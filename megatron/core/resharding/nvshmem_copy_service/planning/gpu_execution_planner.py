# Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.

"""
GPU execution planning for pack/unpack operations.

Converts high-level task descriptions into GPU-ready metadata
(pointer arrays, sizes, chunking) for kernel execution.
"""

from typing import Dict, List, Optional, Tuple

try:
    import cupy as cp

    HAVE_CUPY = True
except ImportError:
    HAVE_CUPY = False

import torch

from ..logger import PELogger
from ..memory.tensor_pointer_utils import TensorPointerExtractor
from ..nvshmem_types import ReceiveRequest, ScheduledBatch


class GPUExecutionPlanner:
    """Plans GPU kernel execution by building pointer arrays and metadata."""

    def __init__(self):
        self.tensor_utils = TensorPointerExtractor()
        self.CHUNK_SIZE = 128 * 1024  # 128KB chunks

    def create_gpu_plans(
        self,
        iter_schedules: List[Dict[str, Optional[ScheduledBatch]]],
        send_slots: List,
        recv_slots: List,
        receive_requests: List[ReceiveRequest],
    ) -> None:
        """
        Build GPU execution plans for all iterations.

        Modifies iter_schedules in-place by adding gpu_plan to each batch.

        Args:
            iter_schedules: List of iteration schedules (dicts with 'send' and 'recv')
            send_slots: List of send buffer slots
            recv_slots: List of receive buffer slots
            receive_requests: List of all receive requests for matching
        """
        if not HAVE_CUPY:
            raise RuntimeError(
                "cupy is not available. Please install cupy to use GPUExecutionPlanner."
            )

        PELogger.debug(f"Creating GPU plans for {len(iter_schedules)} iterations")
        for i, sched in enumerate(iter_schedules):
            send_batch = sched["send"]
            if send_batch:
                # Build Pack Metadata
                ptrs: List[int] = []
                positions: List[int] = []
                sizes: List[int] = []

                for t in send_batch.tasks:
                    # Extract pointer from tensor
                    ptr = self.tensor_utils.get_pointer(t.src_tensor)
                    ptrs.append(ptr)
                    positions.append(t.src_pos)
                    sizes.append(t.size)

                # Plan kernel args for packing
                send_batch.gpu_plan = self._plan_kernel_args(
                    ptrs, positions, sizes, is_pack=True, buffer_base=send_slots[i % 2].data_ptr()
                )
                task_ids = [t.task_id for t in send_batch.tasks]
                PELogger.debug(
                    f"  Iter {i} send plan: {len(send_batch.tasks)} tasks → "
                    f"PE {send_batch.dest_pe}, {send_batch.total_size} bytes"
                )
                displayed_ids = task_ids[:10] if len(task_ids) <= 10 else task_ids[:10] + ["..."]
                PELogger.debug(f"    Send task IDs: {displayed_ids}")

            recv_batch = sched["recv"]
            if recv_batch:
                # Build Unpack Metadata
                summary = recv_batch.tasks_summary

                # Skip if no summary available (shouldn't happen in normal operation)
                if summary is None:
                    PELogger.error(
                        f"Iter {i}: recv batch from PE {recv_batch.src_pe} has no "
                        "tasks_summary - UNPACK WILL BE SKIPPED!"
                    )
                    recv_batch.gpu_plan = None
                    continue

                PELogger.debug(
                    f"  Iter {i} recv from PE {recv_batch.src_pe}: "
                    f"{len(summary.task_ids)} tasks in summary"
                )

                ptrs = []
                positions = []
                sizes = []

                # Create fast lookup map for receive requests
                relevant_reqs: Dict[int, ReceiveRequest] = {
                    r.task_id: r for r in receive_requests if r.src_pe == recv_batch.src_pe
                }

                # Match summary tasks with receive requests
                matched_task_ids: List[int] = []
                unmatched_task_ids: List[int] = []
                for t_id, t_size in zip(summary.task_ids, summary.task_sizes):
                    if t_id in relevant_reqs:
                        req = relevant_reqs[t_id]
                        ptr = self.tensor_utils.get_pointer(req.dest_tensor)
                        ptrs.append(ptr)
                        positions.append(req.dest_pos)
                        sizes.append(t_size)  # Use sender's size
                        matched_task_ids.append(t_id)
                    else:
                        unmatched_task_ids.append(t_id)
                        PELogger.error(
                            f"Iter {i}: Unexpected task {t_id} from PE "
                            f"{recv_batch.src_pe} - no matching recv request!"
                        )

                if unmatched_task_ids:
                    PELogger.error(
                        f"  Iter {i}: {len(unmatched_task_ids)} unmatched tasks "
                        f"from PE {recv_batch.src_pe}: {unmatched_task_ids[:10]}"
                    )

                # Plan kernel args for unpacking
                recv_batch.gpu_plan = self._plan_kernel_args(
                    ptrs, positions, sizes, is_pack=False, buffer_base=recv_slots[i % 2].data_ptr()
                )

                if recv_batch.gpu_plan is None:
                    PELogger.error(
                        f"  Iter {i} recv plan: FAILED - no gpu_plan created for "
                        f"{len(sizes)} tasks from PE {recv_batch.src_pe}"
                    )
                else:
                    PELogger.debug(
                        f"  Iter {i} recv plan: {len(sizes)} tasks ← "
                        f"PE {recv_batch.src_pe}, {recv_batch.total_size} bytes"
                    )
                    displayed_recv_ids = (
                        matched_task_ids[:10]
                        if len(matched_task_ids) <= 10
                        else matched_task_ids[:10] + ["..."]
                    )
                    PELogger.debug(f"    Recv task IDs: {displayed_recv_ids}")

    def _plan_kernel_args(
        self,
        ptrs: List[int],
        positions: List[int],
        sizes: List[int],
        is_pack: bool,
        buffer_base: int,
    ) -> Optional[Tuple[object, object, object, int]]:
        """
        Generate GPU-ready pointer arrays for kernel execution.

        Applies 128KB chunking to break large transfers into smaller pieces.

        Args:
            ptrs: List of tensor data pointers
            positions: List of positions within tensors
            sizes: List of transfer sizes
            is_pack: True for pack (user->buffer), False for unpack (buffer->user)
            buffer_base: Base pointer of the buffer

        Returns:
            Tuple of (cp_src_addrs, cp_dst_addrs, cp_sizes, num_chunks) as
            CuPy arrays, or None if no work.
        """
        h_src_addrs: List[int] = []
        h_dst_addrs: List[int] = []
        h_sizes: List[int] = []

        packed_offset = 0

        for ptr, pos, size in zip(ptrs, positions, sizes):
            num_chunks = (size + self.CHUNK_SIZE - 1) // self.CHUNK_SIZE

            for c in range(num_chunks):
                chunk_offset = c * self.CHUNK_SIZE
                chunk_size = min(self.CHUNK_SIZE, size - chunk_offset)

                if is_pack:
                    # Pack: user tensor -> buffer
                    h_src_addrs.append(ptr + pos + chunk_offset)
                    h_dst_addrs.append(buffer_base + packed_offset + chunk_offset)
                else:
                    # Unpack: buffer -> user tensor
                    h_src_addrs.append(buffer_base + packed_offset + chunk_offset)
                    h_dst_addrs.append(ptr + pos + chunk_offset)

                h_sizes.append(chunk_size)

            packed_offset += size

        total_chunks = len(h_sizes)
        if total_chunks == 0:
            return None

        # Move to GPU using PyTorch, then convert to CuPy for kernel launching
        d_src_addrs = torch.tensor(h_src_addrs, dtype=torch.int64, device="cuda")
        d_dst_addrs = torch.tensor(h_dst_addrs, dtype=torch.int64, device="cuda")
        d_sizes = torch.tensor(h_sizes, dtype=torch.int64, device="cuda")

        # Convert to CuPy arrays (zero-copy) for kernel launching
        cp_src_addrs = cp.asarray(d_src_addrs)
        cp_dst_addrs = cp.asarray(d_dst_addrs)
        cp_sizes = cp.asarray(d_sizes)

        return (cp_src_addrs, cp_dst_addrs, cp_sizes, total_chunks)

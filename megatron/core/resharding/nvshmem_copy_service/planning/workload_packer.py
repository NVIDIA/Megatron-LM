# Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.

from typing import Dict, List

from ..logger import PELogger
from ..nvshmem_types import MAX_SEGMENT_SIZE, MAX_TASKS_PER_BATCH, SendRequest, WorkloadGroup


class WorkloadPacker:
    """
    Packs individual SendRequests into WorkloadGroups (batches)
    destined for the same PE, respecting size limits.
    """

    def pack_workloads(
        self, send_requests: List[SendRequest], n_pes: int
    ) -> Dict[int, List[WorkloadGroup]]:
        """
        Groups requests by destination PE and packs them into batches.
        Returns a map: dest_pe -> list of batches
        """
        PELogger.debug(f"Packing {len(send_requests)} send requests for {n_pes} PEs")
        workloads: Dict[int, List[WorkloadGroup]] = {}

        # Group requests by destination PE
        tasks_by_dest: Dict[int, List[SendRequest]] = {}
        for req in send_requests:
            tasks_by_dest.setdefault(req.dest_pe, []).append(req)

        # Pack tasks for each destination
        for dest_pe in range(n_pes):
            if dest_pe not in tasks_by_dest:
                workloads[dest_pe] = []
                PELogger.debug(f"  Dest PE {dest_pe}: 0 tasks → 0 batches")
                continue

            tasks = tasks_by_dest[dest_pe]
            workloads[dest_pe] = self._pack_single_destination(tasks, dest_pe)

            if workloads[dest_pe]:
                total_size = sum(b.total_size for b in workloads[dest_pe])
                PELogger.debug(
                    f"  Dest PE {dest_pe}: {len(tasks)} tasks → "
                    f"{len(workloads[dest_pe])} batches, {total_size} bytes total"
                )
            else:
                PELogger.debug(
                    f"  Dest PE {dest_pe}: {len(tasks)} tasks → 0 batches (empty after packing)"
                )

        return workloads

    def _pack_single_destination(
        self, tasks: List[SendRequest], dest_pe: int
    ) -> List[WorkloadGroup]:
        if not tasks:
            return []

        # Sort tasks by size (descending) for better bin packing efficiency
        tasks.sort(key=lambda x: x.size, reverse=True)

        batches: List[WorkloadGroup] = []
        current_batch = WorkloadGroup(dest_pe=dest_pe, tasks=[], total_size=0)

        for task in tasks:
            # Check if adding this task would exceed batch constraints
            would_exceed_size = current_batch.total_size + task.size > MAX_SEGMENT_SIZE
            would_exceed_task_cap = len(current_batch.tasks) >= MAX_TASKS_PER_BATCH

            if (would_exceed_size or would_exceed_task_cap) and current_batch.tasks:
                # Finalize current batch
                batches.append(current_batch)
                task_first_10_string = ", ".join([str(t.task_id) for t in current_batch.tasks[:10]])
                PELogger.debug(
                    f"  Packed batch to PE {dest_pe} idx {len(batches) - 1}: "
                    f"{task_first_10_string}... (total {len(current_batch.tasks)} tasks)"
                )
                # Start new batch
                current_batch = WorkloadGroup(dest_pe=dest_pe, tasks=[], total_size=0)

            # Add task to current batch
            current_batch.tasks.append(task)
            current_batch.total_size += task.size

        # Add final batch if not empty
        if current_batch.tasks:
            batches.append(current_batch)

        return batches

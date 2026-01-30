# Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.

from typing import Dict, List, Tuple

from ..logger import PELogger
from ..nvshmem_types import ScheduledBatch, WorkloadGroup, WorkloadSummary


class CommunicationScheduler:
    """
    Builds a conflict-free, iteration-based schedule for communication.
    Ensures that in any given iteration, a PE is not overloaded.
    """

    def __init__(self):
        self.num_iterations = 0

    def build_schedule(
        self, workloads: Dict[int, List[WorkloadGroup]], my_pe: int, n_pes: int
    ) -> Tuple[Dict[int, List[ScheduledBatch]], Dict[Tuple[int, int, int], WorkloadSummary]]:
        """
        Main scheduling method.
        1. Exchanges workload info with other PEs.
        2. Assigns batches to iterations.
        3. Returns:
           - local schedule (iteration -> list of batches)
           - global workload summaries (key: (src, dest, batch_idx) -> summary)
        """
        total_local_batches = sum(len(groups) for groups in workloads.values())
        PELogger.info(f"Building schedule: {total_local_batches} local batches, {n_pes} PEs")

        # Step 1: Collect all batches across all PE pairs
        PELogger.debug("Collecting batches from all PEs...")
        all_batches = self._collect_all_batches(workloads, my_pe, n_pes)
        PELogger.debug(f"Collected {len(all_batches)} total batches globally")

        # Step 2: Assign batches to iterations using conflict-free algorithm
        PELogger.debug("Assigning batches to iterations...")
        self._assign_iterations(all_batches)
        PELogger.info(f"Schedule built: {self.num_iterations} iterations")

        # Step 3: Exchange detailed workload summaries (Task IDs/Sizes)
        # This is needed for receivers to know what tasks are in each batch
        PELogger.debug("Exchanging workload summaries...")
        global_summaries = self._exchange_workload_summaries(workloads, my_pe, n_pes)
        PELogger.debug(f"Exchanged {len(global_summaries)} workload summaries")

        # Step 4: Build schedule map for this PE
        my_batches = [b for b in all_batches if b.src_pe == my_pe or b.dest_pe == my_pe]
        my_batches.sort(key=lambda x: x.iteration)

        final_schedule: Dict[int, List[ScheduledBatch]] = {}
        for b in my_batches:
            final_schedule.setdefault(b.iteration, []).append(b)

        return final_schedule, global_summaries

    def _collect_all_batches(
        self, workloads: Dict[int, List[WorkloadGroup]], my_pe: int, n_pes: int
    ) -> List[ScheduledBatch]:
        """
        Exchanges batch counts and details with all PEs to build a global view.
        Uses torch.distributed for reliable communication.
        """
        import torch.distributed as dist

        # Build local batch list
        local_batches: List[Tuple[int, int, int]] = []
        for dest_pe, groups in workloads.items():
            if dest_pe == my_pe:
                continue
            for i, _ in enumerate(groups):
                local_batches.append((my_pe, dest_pe, i))  # (src, dest, batch_idx)

        PELogger.debug(f"  Local batch count: {len(local_batches)}")
        PELogger.debug(f"  Local batches: {local_batches}")

        # Gather all batches from all PEs using torch.distributed
        all_batches_list: List[List[Tuple[int, int, int]] | None] = [None] * n_pes
        dist.all_gather_object(all_batches_list, local_batches)

        # Flatten into global batch list
        global_batches: List[ScheduledBatch] = []
        for pe_batches in all_batches_list:
            if pe_batches is None:
                continue
            for src, dest, idx in pe_batches:
                global_batches.append(
                    ScheduledBatch(src_pe=src, dest_pe=dest, batch_index=idx, iteration=-1)
                )

        PELogger.debug(f"  Global batches collected: {len(global_batches)} total")

        # Group by source for readability
        batches_by_src: Dict[int, List[Tuple[int, int]]] = {}
        for b in global_batches:
            batches_by_src.setdefault(b.src_pe, []).append((b.dest_pe, b.batch_index))
        for src_pe in sorted(batches_by_src.keys()):
            PELogger.debug(f"    PE {src_pe} sends to: {batches_by_src[src_pe]}")

        return global_batches

    def _assign_iterations(self, batches: List[ScheduledBatch]):
        self.num_iterations = 0
        batches.sort(key=lambda x: (x.src_pe, x.dest_pe, x.batch_index))

        for batch in batches:
            iteration = 0
            assigned = False
            while not assigned:
                if not self._has_conflict(batch, iteration, batches):
                    batch.iteration = iteration
                    self.num_iterations = max(self.num_iterations, iteration + 1)
                    assigned = True
                    PELogger.debug(
                        f"  Assigned batch ({batch.src_pe} → {batch.dest_pe}, "
                        f"idx={batch.batch_index}) to iteration {iteration}"
                    )
                else:
                    iteration += 1

    def _has_conflict(
        self, batch: ScheduledBatch, iteration: int, all_batches: List[ScheduledBatch]
    ) -> bool:
        for other in all_batches:
            if other.iteration == iteration and other is not batch:
                if other.src_pe == batch.src_pe or other.dest_pe == batch.dest_pe:
                    return True
        return False

    def _exchange_workload_summaries(
        self, workloads: Dict[int, List[WorkloadGroup]], my_pe: int, n_pes: int
    ) -> Dict[Tuple[int, int, int], WorkloadSummary]:
        """
        Exchange detailed workload content using torch.distributed.
        Simple and reliable - no NVSHMEM symmetric memory issues.
        """
        import torch.distributed as dist

        # Build local summaries as a simple dict:
        # (src, dest, batch_idx) -> {total_size, task_ids, task_sizes}
        local_summaries: Dict[Tuple[int, int, int], Dict[str, object]] = {}
        batch_count = 0
        total_tasks = 0

        for dest_pe, groups in workloads.items():
            if dest_pe == my_pe:
                continue
            for batch_idx, group in enumerate(groups):
                key = (my_pe, dest_pe, batch_idx)
                local_summaries[key] = {
                    "total_size": group.total_size,
                    "task_ids": [t.task_id for t in group.tasks],
                    "task_sizes": [t.size for t in group.tasks],
                }
                batch_count += 1
                total_tasks += len(group.tasks)

        PELogger.debug(f"  Local summaries: {batch_count} batches, {total_tasks} tasks")

        # Gather all summaries from all PEs using torch.distributed
        all_summaries_list: List[Dict[Tuple[int, int, int], Dict[str, object]] | None] = [
            None
        ] * n_pes
        dist.all_gather_object(all_summaries_list, local_summaries)

        # Merge into global map
        global_map: Dict[Tuple[int, int, int], WorkloadSummary] = {}
        for pe_summaries in all_summaries_list:
            if pe_summaries is None:
                continue
            for key, data in pe_summaries.items():
                summary = WorkloadSummary(
                    total_size=int(data["total_size"]),
                    task_ids=list(data["task_ids"]),
                    task_sizes=list(data["task_sizes"]),
                )
                global_map[key] = summary

        PELogger.debug(f"  Exchanged {len(global_map)} workload summaries")
        return global_map

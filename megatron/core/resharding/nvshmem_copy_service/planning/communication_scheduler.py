# Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.

from typing import Dict, List, Tuple

from ..logger import PELogger
from ..nvshmem_types import ScheduledBatch, WorkloadGroup, WorkloadSummary


class CommunicationScheduler:
    """
    Builds a conflict-free, iteration-based schedule for communication.
    Ensures that in any given iteration, a PE is not overloaded.
    """

    def __init__(self, algorithm: str = "dsatur"):
        """
        Initialize scheduler.

        Args:
            algorithm: Scheduling algorithm to use
                - "greedy": Simple greedy first-fit (baseline)
                - "dsatur": DSatur graph coloring (near-optimal, default)
        """
        self.num_iterations = 0
        self.algorithm = algorithm
        if algorithm not in ["greedy", "dsatur"]:
            raise ValueError(f"Unknown algorithm: {algorithm}. Use 'greedy' or 'dsatur'")

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
        PELogger.debug(f"Assigning batches to iterations using '{self.algorithm}' algorithm...")
        if self.algorithm == "dsatur":
            self._assign_iterations_dsatur(all_batches)
        elif self.algorithm == "greedy":
            self._assign_iterations_greedy(all_batches)
        PELogger.info(f"Schedule built ({self.algorithm}): {self.num_iterations} iterations")

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
        # NOTE: all_gather_object is a collective operation that requires all ranks
        # to participate, even those with empty workloads
        PELogger.debug(f"  About to call all_gather_object with {n_pes} PEs...")
        all_batches_list: List[List[Tuple[int, int, int]] | None] = [None] * n_pes
        dist.all_gather_object(all_batches_list, local_batches)
        PELogger.debug(f"  all_gather_object completed successfully")

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

    def _assign_iterations_greedy(self, batches: List[ScheduledBatch]):
        """
        BASELINE: Greedy first-fit scheduling algorithm.

        Assigns batches to iterations using simple greedy first-fit.
        Processes batches in sorted order and assigns each to the first
        available iteration with no conflicts.

        Fast but suboptimal - serves as baseline for comparison.
        """
        self.num_iterations = 0

        # Sort batches: process batches with more potential conflicts first
        # This heuristic (largest-degree-first) often produces better colorings
        batches.sort(key=lambda x: (x.src_pe, x.dest_pe, x.batch_index))

        # Track which PEs are busy (sending or receiving) in each iteration
        # iteration -> {src_pes: set, dst_pes: set}
        iteration_usage = []

        for batch in batches:
            # Find first iteration where this batch fits (no conflicts)
            assigned = False
            for iter_idx in range(len(iteration_usage)):
                # Check if src_pe or dest_pe are already busy in this iteration
                if (batch.src_pe not in iteration_usage[iter_idx]['src_pes'] and
                    batch.src_pe not in iteration_usage[iter_idx]['dst_pes'] and
                    batch.dest_pe not in iteration_usage[iter_idx]['src_pes'] and
                    batch.dest_pe not in iteration_usage[iter_idx]['dst_pes']):
                    # No conflict - assign to this iteration
                    batch.iteration = iter_idx
                    iteration_usage[iter_idx]['src_pes'].add(batch.src_pe)
                    iteration_usage[iter_idx]['dst_pes'].add(batch.dest_pe)
                    assigned = True
                    PELogger.debug(
                        f"  Assigned batch ({batch.src_pe} → {batch.dest_pe}, "
                        f"idx={batch.batch_index}) to iteration {iter_idx}"
                    )
                    break

            if not assigned:
                # Need a new iteration
                new_iter = len(iteration_usage)
                batch.iteration = new_iter
                iteration_usage.append({
                    'src_pes': {batch.src_pe},
                    'dst_pes': {batch.dest_pe}
                })
                PELogger.debug(
                    f"  Assigned batch ({batch.src_pe} → {batch.dest_pe}, "
                    f"idx={batch.batch_index}) to NEW iteration {new_iter}"
                )

        self.num_iterations = len(iteration_usage)
        PELogger.info(f"Greedy scheduling: {len(batches)} batches → {self.num_iterations} iterations")

    def _assign_iterations_dsatur(self, batches: List[ScheduledBatch]):
        """
        DSatur (Saturation Degree) algorithm for near-optimal scheduling.

        Colors batches (assigns to iterations) by prioritizing batches with
        the highest saturation degree (most constrained choices).

        This is the standard algorithm for graph coloring and produces
        near-optimal results in practice (typically 2-3x better than greedy).
        """
        if not batches:
            self.num_iterations = 0
            return

        PELogger.info(f"DSatur scheduling: {len(batches)} batches")

        # Step 1: Build conflict graph
        conflicts = self._build_conflict_graph(batches)

        # Step 2: DSatur algorithm
        uncolored = set(range(len(batches)))
        colors = {}  # batch_idx -> iteration (color)

        # Start with batch that has most conflicts (highest degree)
        first_batch = max(range(len(batches)), key=lambda i: len(conflicts[i]))
        colors[first_batch] = 0
        uncolored.remove(first_batch)
        PELogger.debug(f"  Colored batch {first_batch} with color 0 (highest degree: {len(conflicts[first_batch])} conflicts)")

        # Color remaining batches
        while uncolored:
            # Select batch with highest saturation degree
            # Tie-break by degree (most conflicts), then by index
            best = max(uncolored, key=lambda i: (
                self._saturation_degree(i, colors, conflicts),
                len(conflicts[i]),
                -i  # Negative for stable ordering
            ))

            # Assign smallest available color (iteration)
            neighbor_colors = {colors[j] for j in conflicts[best] if j in colors}
            color = 0
            while color in neighbor_colors:
                color += 1

            colors[best] = color
            uncolored.remove(best)

            if len(uncolored) % 20 == 0:  # Log progress periodically
                PELogger.debug(f"  Colored {len(colors)}/{len(batches)} batches, using {max(colors.values())+1} iterations so far")

        # Step 3: Apply colors to batches
        for idx, batch in enumerate(batches):
            batch.iteration = colors[idx]

        self.num_iterations = max(colors.values()) + 1
        PELogger.info(f"DSatur result: {len(batches)} batches → {self.num_iterations} iterations")

    def _build_conflict_graph(self, batches: List[ScheduledBatch]) -> List[set]:
        """
        Build conflict graph: conflicts[i] = set of batch indices that conflict with i.

        Two batches conflict if they share a source PE or destination PE.
        A PE cannot send and receive simultaneously in the same iteration.
        """
        n = len(batches)
        conflicts = [set() for _ in range(n)]

        PELogger.debug(f"Building conflict graph for {n} batches...")

        for i in range(n):
            for j in range(i + 1, n):
                # Check if batches i and j conflict
                if (batches[i].src_pe == batches[j].src_pe or
                    batches[i].src_pe == batches[j].dest_pe or
                    batches[i].dest_pe == batches[j].src_pe or
                    batches[i].dest_pe == batches[j].dest_pe):
                    conflicts[i].add(j)
                    conflicts[j].add(i)

        total_edges = sum(len(c) for c in conflicts) // 2
        max_degree = max(len(c) for c in conflicts) if conflicts else 0
        avg_degree = sum(len(c) for c in conflicts) / n if n > 0 else 0
        PELogger.debug(f"Conflict graph: {total_edges} edges, max degree: {max_degree}, avg degree: {avg_degree:.1f}")

        return conflicts

    def _saturation_degree(self, batch_idx: int, colors: dict, conflicts: List[set]) -> int:
        """
        Saturation degree = number of distinct colors used by neighbors.

        Higher saturation means the batch has fewer color choices available,
        so it should be colored earlier (more constrained).
        """
        neighbor_colors = {colors[j] for j in conflicts[batch_idx] if j in colors}
        return len(neighbor_colors)

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
        # NOTE: all_gather_object is a collective operation that requires all ranks
        # to participate, even those with empty workloads
        PELogger.debug(f"  About to call all_gather_object for summaries...")
        all_summaries_list: List[Dict[Tuple[int, int, int], Dict[str, object]] | None] = [
            None
        ] * n_pes
        dist.all_gather_object(all_summaries_list, local_summaries)
        PELogger.debug(f"  all_gather_object for summaries completed successfully")

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

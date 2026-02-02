# Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.

"""
Remote Copy Service - Main orchestrator for NVSHMEM-based GPU-to-GPU transfers.

This service coordinates task segmentation, workload packing, scheduling,

GPU resource management, and pipelined execution.
"""

from typing import Dict, List, Optional, Tuple

try:
    import nvshmem.core

    HAVE_NVSHMEM = True
except ImportError:
    HAVE_NVSHMEM = False

import torch.cuda.nvtx as nvtx

from .core import GPUResourceManager, KernelLauncher, PipelineExecutor
from .logger import PELogger
from .memory import DoubleBufferManager
from .nvshmem_types import ReceiveRequest, ScheduledBatch, SendRequest, WorkloadSummary
from .planning import CommunicationScheduler, GPUExecutionPlanner, TaskSegmenter, WorkloadPacker


class RemoteCopyService:
    """
    Main service for managing remote GPU-to-GPU data transfers.

    Provides high-level API for registering transfers, scheduling,
    and executing pipelined communication with NVSHMEM.
    """

    def __init__(self):
        # Core components
        self.gpu_resources = GPUResourceManager()
        self.buffer_manager = DoubleBufferManager()
        self.kernel_launcher = KernelLauncher()
        self.pipeline_executor = None  # Created after init

        # Planning components
        self.task_segmenter = TaskSegmenter()
        self.workload_packer = WorkloadPacker()
        self.comm_scheduler = CommunicationScheduler()
        self.gpu_planner = GPUExecutionPlanner()

        # State
        self.send_requests: List[SendRequest] = []
        self.receive_requests: List[ReceiveRequest] = []
        self.iter_schedules: Optional[List[Dict]] = None
        self.num_iterations: int = 0

        # Events for double-buffering
        self.pack_events = []
        self.unpack_events = []

    @property
    def my_pe(self) -> int:
        """Get this PE's rank."""
        return self.gpu_resources.my_pe

    @property
    def n_pes(self) -> int:
        """Get total number of PEs."""
        return self.gpu_resources.n_pes

    @property
    def device(self):
        """Get CUDA device."""
        return self.gpu_resources.device

    @property
    def initialized(self) -> bool:
        """Check if service is initialized."""
        return self.gpu_resources.initialized

    def init(self, log_level: str = "INFO") -> None:
        """
        Initialize the service.

        Sets up NVSHMEM, CUDA device, streams, buffers, and kernels.
        Expects to be launched with torchrun.

        Args:
            log_level: Logging level (TRACE, DEBUG, INFO, WARN, ERROR)
        """
        if not HAVE_NVSHMEM:
            raise RuntimeError(
                "nvshmem.core is not available. Please install nvshmem to use NVSHMEMCopyService."
            )

        # Initialize GPU resources (NVSHMEM, device, streams)
        self.gpu_resources.init()

        # Initialize logger after PE ID is known
        PELogger.init(self.my_pe, level=log_level)
        PELogger.info(f"Initializing RemoteCopyService on PE {self.my_pe}/{self.n_pes}")

        # Allocate double-buffered send/recv slots
        self.buffer_manager.allocate()
        PELogger.debug("Allocated double-buffered send/recv slots")

        # Load CUDA kernels
        self.kernel_launcher.load_kernels()
        PELogger.debug("Loaded CUDA kernels")

        # Cache CuPy stream wrappers for efficient kernel launching
        self.kernel_launcher.set_streams(
            self.gpu_resources.pack_stream, self.gpu_resources.unpack_stream
        )
        PELogger.debug("Cached CuPy stream wrappers")

        # Create pipeline executor with dependencies
        self.pipeline_executor = PipelineExecutor(
            self.kernel_launcher, self.buffer_manager, self.my_pe
        )

        # Set streams on pipeline executor
        self.pipeline_executor.set_streams(
            self.gpu_resources.pack_stream,
            self.gpu_resources.unpack_stream,
            self.gpu_resources.send_stream,
            self.gpu_resources.copy_stream,
            self.gpu_resources.torch_pack_stream,
            self.gpu_resources.torch_unpack_stream,
            self.gpu_resources.torch_copy_stream,
        )
        PELogger.info("Initialization complete")

    def register_send(
        self, task_id: int, src_tensor, src_pos: int, size: int, dest_pe: int
    ) -> None:
        """
        Register a send operation.

        Args:
            task_id: Unique task identifier
            src_tensor: Source tensor (PyTorch/CuPy tensor or pointer)
            src_pos: Starting position in source tensor
            size: Number of bytes to send
            dest_pe: Destination PE rank
        """
        if dest_pe >= self.n_pes or dest_pe < 0:
            PELogger.error(f"Error: Invalid destination PE {dest_pe}")
            return

        req = SendRequest(task_id, src_tensor, src_pos, size, dest_pe)
        self.send_requests.append(req)

    def register_receive(
        self, task_id: int, dest_tensor, dest_pos: int, size: int, src_pe: int
    ) -> None:
        """
        Register a receive operation.

        Args:
            task_id: Unique task identifier
            dest_tensor: Destination tensor (PyTorch/CuPy tensor or pointer)
            dest_pos: Starting position in destination tensor
            size: Number of bytes to receive
            src_pe: Source PE rank
        """
        if src_pe >= self.n_pes or src_pe < 0:
            PELogger.error(f"Error: Invalid source PE {src_pe}")
            return

        req = ReceiveRequest(task_id, dest_tensor, dest_pos, size, src_pe)
        self.receive_requests.append(req)

    def schedule(self) -> None:
        """
        Build execution schedule.

        Can be called once and followed by multiple run() calls for
        repeated execution with the same communication pattern.

        Steps:
        1. Segment large tasks into manageable chunks
        2. Pack tasks into batches
        3. Schedule batches to iterations (conflict-free)
        4. Build GPU execution plans (pointer arrays, chunking)
        5. Create synchronization events
        """
        if not self.initialized:
            raise RuntimeError("RemoteCopyService not initialized")

        PELogger.info(
            f"Starting schedule: {len(self.send_requests)} send requests, "
            f"{len(self.receive_requests)} receive requests"
        )

        # Step 1: Segment tasks (break large tasks into chunks)
        PELogger.debug("Step 1: Segmenting tasks...")
        orig_send_count = len(self.send_requests)
        orig_recv_count = len(self.receive_requests)
        self._segment_tasks()
        PELogger.info(
            f"Segmented: {orig_send_count} sends → {len(self.send_requests)} segments, "
            f"{orig_recv_count} recvs → {len(self.receive_requests)} segments"
        )

        # Step 2: Pack tasks into workload groups
        PELogger.debug("Step 2: Packing workloads...")
        workloads = self.workload_packer.pack_workloads(self.send_requests, self.n_pes)
        total_batches = sum(len(batches) for batches in workloads.values())
        active_pes = sum(1 for batches in workloads.values() if batches)
        PELogger.info(f"Packed: {total_batches} batches across {active_pes} destination PEs")

        # Step 3: Schedule workloads to iterations
        PELogger.debug("Step 3: Building communication schedule...")
        schedule, global_summaries = self.comm_scheduler.build_schedule(
            workloads, self.my_pe, self.n_pes
        )

        self.num_iterations = self.comm_scheduler.num_iterations
        PELogger.info(f"Scheduled: {total_batches} batches → {self.num_iterations} iterations")

        # Step 4: Prepare iteration schedules
        PELogger.debug("Step 4: Preparing iteration schedules...")
        self.iter_schedules = self._prepare_iter_schedules(
            schedule, workloads, global_summaries, self.num_iterations
        )

        # Step 5: Build GPU execution plans
        PELogger.debug("Step 5: Building GPU execution plans...")
        self.gpu_planner.create_gpu_plans(
            self.iter_schedules,
            self.buffer_manager.send_slots,
            self.buffer_manager.recv_slots,
            self.receive_requests,
        )

        # Step 6: Create double-buffered events
        PELogger.debug("Step 6: Creating synchronization events...")
        self.pack_events, self.unpack_events = self.gpu_resources.create_events(num_events=2)
        self.pipeline_executor.set_events(self.pack_events, self.unpack_events)

        PELogger.info(f"Schedule complete: {self.num_iterations} iterations ready")

    def run(self) -> None:
        """
        Execute the scheduled communication.

        Can be called multiple times after a single schedule() call
        to repeat the same communication pattern.
        """
        # import torch
        # torch.save(self.send_requests, f"send_requests_{torch.distributed.get_rank()}.pt")
        # torch.save(self.receive_requests, f"receive_requests_{torch.distributed.get_rank()}.pt")

        if not self.initialized:
            raise RuntimeError("RemoteCopyService not initialized")
        if self.iter_schedules is None:
            raise RuntimeError("Must call schedule() before run()")

        PELogger.info(f"Starting execution: {self.num_iterations} iterations")

        # Start timing
        nvtx.range_push("RemoteCopyService.run_total")

        # Global barrier before execution
        PELogger.debug("Barrier: Synchronizing all PEs before execution")
        nvshmem.core.barrier_all(stream=self.gpu_resources.send_stream)
        self.gpu_resources.send_stream.sync()

        # Execute pipelined communication
        nvtx.range_push("execute_pipeline")
        self.pipeline_executor.execute_pipeline(self.iter_schedules, self.num_iterations)
        nvtx.range_pop()  # execute_pipeline

        # Global barrier after execution
        PELogger.debug("Barrier: Synchronizing all PEs after pipeline")
        nvshmem.core.barrier_all(stream=self.gpu_resources.send_stream)

        # Process same-PE transfers
        self.pipeline_executor.process_self_moves(self.send_requests, self.receive_requests)

        # End timing range
        nvtx.range_pop()  # RemoteCopyService.run_total

    def clear_requests(self) -> None:
        """
        Clear registered requests and schedule.

        Call this before registering a new set of transfers.
        """
        self.send_requests = []
        self.receive_requests = []
        self.iter_schedules = None
        self.num_iterations = 0
        self.pack_events = []
        self.unpack_events = []

    def finalize(self) -> None:
        """Cleanup resources."""
        PELogger.info("Finalizing RemoteCopyService")

        # Barrier to ensure all PEs are ready to finalize
        try:
            PELogger.debug("Barrier: Synchronizing all PEs before finalize")
            nvshmem.core.barrier_all(stream=self.gpu_resources.send_stream)
            self.gpu_resources.send_stream.sync()
        except Exception as e:
            PELogger.error(f"Error in final barrier: {e}")

        # Free buffers
        self.buffer_manager.free()

        # Finalize GPU resources (this will call nvshmem.core.finalize internally)
        self.gpu_resources.finalize()

        PELogger.info("RemoteCopyService finalized")
        PELogger.shutdown()

    def _segment_tasks(self) -> None:
        """Segment tasks into manageable chunks."""
        new_sends: List[SendRequest] = []
        for req in self.send_requests:
            segments = self.task_segmenter.segment_send_request(req)
            new_sends.extend(segments)
            if len(segments) > 1:
                PELogger.debug(
                    f"  Segmented send task {req.task_id}: "
                    f"{req.size} bytes → {len(segments)} segments"
                )
        self.send_requests = new_sends

        new_recvs: List[ReceiveRequest] = []
        for req in self.receive_requests:
            segments = self.task_segmenter.segment_receive_request(req)
            new_recvs.extend(segments)
            if len(segments) > 1:
                PELogger.debug(
                    f"  Segmented recv task {req.task_id}: "
                    f"{req.size} bytes → {len(segments)} segments"
                )
        self.receive_requests = new_recvs

    def _prepare_iter_schedules(
        self,
        schedule_batches: Dict[int, List[ScheduledBatch]],
        workloads: Dict[int, List],
        global_summaries: Dict[Tuple[int, int, int], WorkloadSummary],
        num_iterations: int,
    ) -> List[Dict]:
        """
        Organize schedule into iteration-based structure.

        Returns:
            List of dicts with 'send' and 'recv' keys for each iteration
        """
        iter_schedules: List[Dict[str, Optional[ScheduledBatch]]] = []

        for i in range(num_iterations):
            sched: Dict[str, Optional[ScheduledBatch]] = {"send": None, "recv": None}

            if i in schedule_batches:
                batches = schedule_batches[i]

                for b in batches:
                    # Skip same-PE transfers (handled separately by process_self_moves)
                    if b.src_pe == b.dest_pe:
                        PELogger.debug(
                            f"  Iter {i}: Skipping same-PE batch " f"({b.src_pe} → {b.dest_pe})"
                        )
                        continue

                    if b.src_pe == self.my_pe:
                        # This PE sends in this iteration
                        b.tasks = workloads[b.dest_pe][b.batch_index].tasks
                        b.total_size = workloads[b.dest_pe][b.batch_index].total_size
                        sched["send"] = b
                        PELogger.debug(
                            f"  Iter {i}: Send to PE {b.dest_pe}, batch "
                            f"{b.batch_index}, {len(b.tasks)} tasks, "
                            f"{b.total_size} bytes"
                        )

                    elif b.dest_pe == self.my_pe:
                        # This PE receives in this iteration
                        key = (b.src_pe, b.dest_pe, b.batch_index)
                        if key in global_summaries:
                            summary = global_summaries[key]
                            b.tasks_summary = summary
                            b.total_size = summary.total_size
                        else:
                            PELogger.error(
                                f"  Iter {i}: Missing workload summary for "
                                f"recv from PE {b.src_pe}, batch {b.batch_index}"
                            )
                            PELogger.error(
                                "  Available keys in global_summaries: "
                                f"{list(global_summaries.keys())}"
                            )
                            b.tasks_summary = None
                            b.total_size = 0
                        sched["recv"] = b
                        PELogger.debug(
                            f"  Iter {i}: Recv from PE {b.src_pe}, batch "
                            f"{b.batch_index}, {b.total_size} bytes"
                        )

            iter_schedules.append(sched)

        return iter_schedules

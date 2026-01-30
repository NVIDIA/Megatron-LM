# Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.

"""
Pipelined communication execution engine.

Orchestrates the pack/send/unpack pipeline with double-buffering
and proper stream synchronization.
"""

from typing import Dict, List, Optional

try:
    import nvshmem.core

    HAVE_NVSHMEM = True
except ImportError:
    HAVE_NVSHMEM = False

import torch

from ..logger import PELogger
from ..memory.double_buffer_manager import DoubleBufferManager
from ..nvshmem_types import ReceiveRequest, ScheduledBatch, SendRequest
from .kernel_launcher import KernelLauncher


class PipelineExecutor:
    """Executes pipelined NVSHMEM communication with pack/send/unpack overlap."""

    def __init__(
        self, kernel_launcher: KernelLauncher, buffer_manager: DoubleBufferManager, my_pe: int
    ):
        """
        Initialize pipeline executor.

        Args:
            kernel_launcher: KernelLauncher instance for pack/unpack kernels
            buffer_manager: DoubleBufferManager for send/recv buffers
            my_pe: This PE's rank
        """
        self.kernel_launcher = kernel_launcher
        self.buffer_manager = buffer_manager
        self.my_pe = my_pe

        # Streams (will be set by service)
        self.pack_stream = None
        self.unpack_stream = None
        self.send_stream = None
        self.copy_stream = None

        self.torch_pack_stream = None
        self.torch_unpack_stream = None
        self.torch_copy_stream = None

        # Events for double-buffered synchronization
        self.pack_events = []
        self.unpack_events = []

    def set_streams(
        self,
        pack_stream,
        unpack_stream,
        send_stream,
        copy_stream,
        torch_pack_stream,
        torch_unpack_stream,
        torch_copy_stream,
    ):
        """Set CUDA streams for execution."""
        self.pack_stream = pack_stream
        self.unpack_stream = unpack_stream
        self.send_stream = send_stream
        self.copy_stream = copy_stream

        self.torch_pack_stream = torch_pack_stream
        self.torch_unpack_stream = torch_unpack_stream
        self.torch_copy_stream = torch_copy_stream

    def set_events(self, pack_events: List, unpack_events: List):
        """Set double-buffered CUDA events."""
        self.pack_events = pack_events
        self.unpack_events = unpack_events

    def execute_pipeline(
        self, iter_schedules: List[Dict[str, Optional[ScheduledBatch]]], num_iterations: int
    ) -> None:
        """
        Execute pipelined communication.

        Pipeline stages:
        1. Pack NEXT iteration (async)
        2. Unpack PRIOR iteration (async)
        3. Send CURRENT iteration (sync)
        4. Barrier
        5. Wait for async pack/unpack to complete

        Args:
            iter_schedules: List of iteration schedules
            num_iterations: Total number of iterations
        """
        PELogger.info(f"Executing pipeline: {num_iterations} iterations")

        # Priming: Pack iteration 0 and WAIT for completion
        if num_iterations > 0 and iter_schedules[0]["send"]:
            torch.cuda.nvtx.range_push("Priming")
            PELogger.debug("Priming: Packing iteration 0")
            self._launch_pack(0, iter_schedules[0]["send"])
            self.pack_events[0].synchronize()
            torch.cuda.nvtx.range_pop()

        for i in range(num_iterations):
            torch.cuda.nvtx.range_push(f"Iteration {i}")
            has_send = iter_schedules[i]["send"] is not None
            has_recv = iter_schedules[i]["recv"] is not None
            has_next_send = i + 1 < num_iterations and iter_schedules[i + 1]["send"] is not None
            has_prior_recv = i > 0 and iter_schedules[i - 1]["recv"] is not None

            slot = i % 2

            # Log iteration start
            send_info = (
                f" → PE {iter_schedules[i]['send'].dest_pe} "
                f"({iter_schedules[i]['send'].total_size} bytes)"
                if has_send
                else ""
            )
            recv_info = (
                f" ← PE {iter_schedules[i]['recv'].src_pe} "
                f"({iter_schedules[i]['recv'].total_size} bytes)"
                if has_recv
                else ""
            )
            PELogger.debug(f"Iteration {i}/{num_iterations}: slot={slot}{send_info}{recv_info}")

            # Step 1: Pack NEXT iteration (async)
            if has_next_send:
                torch.cuda.nvtx.range_push("Step 1: Pack Next")
                next_batch = iter_schedules[i + 1]["send"]
                assert next_batch is not None
                PELogger.debug(
                    f"  Pack next (iter {i+1}): {len(next_batch.tasks)} tasks "
                    f"→ PE {next_batch.dest_pe}"
                )
                self._launch_pack(i + 1, next_batch)
                torch.cuda.nvtx.range_pop()

            # Step 2: Unpack PRIOR iteration (async)
            if has_prior_recv:
                torch.cuda.nvtx.range_push("Step 2: Unpack Prior")
                prior_batch = iter_schedules[i - 1]["recv"]
                assert prior_batch is not None
                PELogger.debug(
                    f"  Unpack prior (iter {i-1}): {prior_batch.total_size} bytes "
                    f"← PE {prior_batch.src_pe}"
                )
                self._launch_unpack(i - 1, prior_batch)
                torch.cuda.nvtx.range_pop()

            # Step 3: Send CURRENT iteration
            if has_send:
                torch.cuda.nvtx.range_push("Step 3: Send Current")
                batch = iter_schedules[i]["send"]
                assert batch is not None
                transfer_size = batch.total_size
                PELogger.debug(f"  Send current: {transfer_size} bytes → PE {batch.dest_pe}")

                nvshmem.core.put(
                    self.buffer_manager.recv_slots[slot][0:transfer_size],
                    self.buffer_manager.send_slots[slot][0:transfer_size],
                    batch.dest_pe,
                    stream=self.send_stream,
                )
                torch.cuda.nvtx.range_pop()

            # Ensure send completes
            self.send_stream.sync()
            nvshmem.core.quiet(stream=self.send_stream)

            # Step 4: Global barrier
            torch.cuda.nvtx.range_push("Step 4: Barrier")
            nvshmem.core.barrier_all(stream=self.send_stream)
            self.send_stream.sync()
            torch.cuda.nvtx.range_pop()

            # Step 5: Wait for async pack/unpack to complete
            torch.cuda.nvtx.range_push("Step 5: Wait Async")
            if has_prior_recv:
                self.unpack_events[(i - 1) % 2].synchronize()
            if has_next_send:
                self.pack_events[(i + 1) % 2].synchronize()
            torch.cuda.nvtx.range_pop()

            torch.cuda.nvtx.range_pop()

        # Final unpack for last iteration
        if num_iterations > 0 and iter_schedules[num_iterations - 1]["recv"]:
            torch.cuda.nvtx.range_push("Final Unpack")
            PELogger.debug(f"Final unpack: iteration {num_iterations-1}")
            last_recv = iter_schedules[num_iterations - 1]["recv"]
            assert last_recv is not None
            self._launch_unpack(num_iterations - 1, last_recv)
            self.unpack_events[(num_iterations - 1) % 2].synchronize()
            torch.cuda.nvtx.range_pop()

        PELogger.info(f"Pipeline complete: {num_iterations} iterations")

    def _launch_pack(self, iteration: int, batch: ScheduledBatch) -> None:
        """Launch pack kernel for given iteration."""
        if not batch.gpu_plan:
            return

        self.kernel_launcher.launch_pack(
            batch.gpu_plan,
            self.pack_stream,
            self.torch_pack_stream,
            self.pack_events[iteration % 2],
        )

    def _launch_unpack(self, iteration: int, batch: ScheduledBatch) -> None:
        """Launch unpack kernel for given iteration."""
        if not batch.gpu_plan:
            return

        self.kernel_launcher.launch_unpack(
            batch.gpu_plan,
            self.unpack_stream,
            self.torch_unpack_stream,
            self.unpack_events[iteration % 2],
        )

    def process_self_moves(
        self, send_requests: List[SendRequest], receive_requests: List[ReceiveRequest]
    ) -> None:
        """
        Handle same-PE transfers (where src_pe == dest_pe == my_pe).

        Uses PyTorch copy on the copy stream for efficiency.

        Args:
            send_requests: List of send requests
            receive_requests: List of receive requests
        """
        # Match send/recv requests where src_pe == dest_pe == my_pe
        local_sends = {r.task_id: r for r in send_requests if r.dest_pe == self.my_pe}
        local_recvs = [r for r in receive_requests if r.src_pe == self.my_pe]

        if local_recvs:
            PELogger.debug(f"Processing {len(local_recvs)} self-moves")

        num_processed = 0
        with torch.cuda.stream(self.torch_copy_stream):
            for recv_req in local_recvs:
                if recv_req.task_id in local_sends:
                    send_req = local_sends[recv_req.task_id]
                    PELogger.debug(
                        "  Self-move: task_id=%d, size=%d bytes", recv_req.task_id, send_req.size
                    )

                    # Create views of the tensors with offsets
                    src_view = send_req.src_tensor[
                        send_req.src_pos : send_req.src_pos + send_req.size
                    ]
                    dest_view = recv_req.dest_tensor[
                        recv_req.dest_pos : recv_req.dest_pos + send_req.size
                    ]

                    # Async copy on the copy stream
                    dest_view.copy_(src_view, non_blocking=True)
                    num_processed += 1

        # Synchronize the PyTorch stream
        self.torch_copy_stream.synchronize()

        if num_processed > 0:
            PELogger.info("Self-moves complete: %d transfers", num_processed)

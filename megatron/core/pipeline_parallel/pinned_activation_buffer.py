# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Bounded pinned storage for fine-grained activation offloading."""

import math

import torch


def _cuda_runtime():
    """Return the cuda-python runtime bindings."""
    try:
        import cuda.bindings.runtime as cuda_runtime  # type: ignore[import-untyped]
    except ImportError as error:
        raise RuntimeError("A fixed pinned activation buffer requires cuda-python") from error
    return cuda_runtime


class PinnedActivationBuffer:
    """Pack saved activations into one fixed, reusable pinned buffer.

    Allocations append within the buffer. Once all backups have been reloaded,
    the next allocation reuses the buffer after waiting for its H2D readers.
    Interleaved chunks share one budget; storage is not reclaimed until all
    outstanding backups are consumed. No individual-tensor free list is needed.

    Captured addresses remain reserved for this object's lifetime: replay does
    not execute Python allocation/release bookkeeping. Each capture owns a
    separate range, reusable within that capture after its readers finish.
    Eager epochs use the remaining tail and cannot overwrite graph backups.

    Capacity must be a power of two so the host allocator cannot round the
    backing allocation beyond the budget. Storage is allocated at construction
    and stays alive with this object, including while captured graphs retain
    its addresses. Exhaustion raises an error; the buffer never grows.
    """

    ALIGNMENT = 256

    def __init__(self, capacity_bytes: int) -> None:
        if capacity_bytes < self.ALIGNMENT or capacity_bytes & (capacity_bytes - 1):
            raise ValueError(
                "Pinned activation buffer capacity must be a power of two and at least 256 bytes"
            )
        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError("Create the pinned activation buffer before CUDA graph capture")
        self.capacity_bytes = capacity_bytes
        self.peak_live_bytes = 0
        self.live_bytes = 0
        self._storage = torch.empty(
            capacity_bytes, dtype=torch.uint8, device="cpu", pin_memory=True
        )
        self._typed_storage: dict[torch.dtype, torch.Tensor] = {}
        self._offset = 0
        self.graph_reserved_bytes = 0
        self._capture_id: int | None = None
        self._capture_start = 0
        self._live: dict[int, int] = {}
        self._reader_streams: dict[int, torch.cuda.Stream] = {}
        self._pending_events: list[torch.cuda.Event] = []
        self._captured_events: list[torch.cuda.Event] = []
        self._writers_waited: set[int] = set()

    def allocate(
        self, shape: tuple, dtype: torch.dtype, *, stream: torch.cuda.Stream | None = None
    ) -> torch.Tensor:
        """Allocate a view whose first write uses the current stream.

        Pass the already-active stream to avoid rediscovering it for every tensor.
        This does not switch streams; the caller still enqueues the copy on it.
        """
        if stream is None:
            stream = torch.cuda.current_stream(torch.cuda.current_device())
        capturing = torch.cuda.is_current_stream_capturing()
        epoch_start = self.graph_reserved_bytes
        if capturing:
            cuda_runtime = _cuda_runtime()
            error, _, capture_id, *_ = cuda_runtime.cudaStreamGetCaptureInfo(stream.cuda_stream)
            if error != cuda_runtime.cudaError_t.cudaSuccess:
                raise RuntimeError(f"Cannot identify pinned buffer capture: {error}")
            if capture_id != self._capture_id:
                self._capture_id = capture_id
                self._capture_start = max(epoch_start, self._offset if self._live else 0)
            epoch_start = self._capture_start
        if not self._live:
            self._writers_waited.clear()
            self._offset = epoch_start
        else:
            self._offset = max(self._offset, epoch_start)
        if stream.cuda_stream not in self._writers_waited:
            if capturing:
                # External event nodes retain CUDA handles, not Python owners.
                # Keep every captured wait/record handle alive with the buffer.
                self._captured_events.extend(self._pending_events)
            for event in self._pending_events:
                stream.wait_event(event)
            self._writers_waited.add(stream.cuda_stream)
        itemsize = dtype.itemsize
        nbytes = math.prod(shape) * itemsize
        aligned = ((nbytes + self.ALIGNMENT - 1) & -self.ALIGNMENT) or self.ALIGNMENT
        if self.capacity_bytes - self._offset < aligned:
            raise RuntimeError(
                "Pinned activation buffer budget exceeded: "
                f"request={aligned} bytes, capacity={self.capacity_bytes} bytes, "
                f"live={self.live_bytes} bytes, "
                f"graph_reserved={self.graph_reserved_bytes} bytes, "
                f"unused_tail={self.capacity_bytes - self._offset} bytes. "
                "Increase fine_grained_offloading_buffer_size_gib or reduce offloaded activations."
            )
        start = self._offset
        typed_storage = self._typed_storage.get(dtype)
        if typed_storage is None:
            typed_storage = self._storage.view(dtype)
            self._typed_storage[dtype] = typed_storage
        strides = [1] * len(shape)
        for axis in range(len(shape) - 2, -1, -1):
            strides[axis] = strides[axis + 1] * max(shape[axis + 1], 1)
        tensor = typed_storage.as_strided(shape, strides, start // itemsize)
        self._offset += aligned
        if capturing:
            self.graph_reserved_bytes = max(self.graph_reserved_bytes, self._offset)
        self._live[id(tensor)] = nbytes
        self.live_bytes += nbytes
        self.peak_live_bytes = max(self.peak_live_bytes, self.live_bytes)
        return tensor

    def release(self, tensor: torch.Tensor, *, stream: torch.cuda.Stream | None = None) -> None:
        """Retire a backup after its final H2D read is enqueued on the active stream."""
        if torch.cuda.is_current_stream_capturing():
            # Even a backup allocated eagerly can be read by a captured graph.
            # Capture retires its logical use, not the graph's address ownership.
            end = (
                tensor.storage_offset() * tensor.element_size()
                + tensor.numel() * tensor.element_size()
            )
            end = (end + self.ALIGNMENT - 1) & -self.ALIGNMENT
            self.graph_reserved_bytes = max(self.graph_reserved_bytes, end)
        self.live_bytes -= self._live.pop(id(tensor))
        if stream is None:
            stream = torch.cuda.current_stream(torch.cuda.current_device())
        self._reader_streams[stream.cuda_stream] = stream
        if not self._live:
            # Events protect reuse across streams and across iterations. External
            # events also remain observable after a captured graph is replayed.
            self._pending_events = []
            for reader in self._reader_streams.values():
                event = torch.cuda.Event(external=True)
                event.record(reader)
                self._pending_events.append(event)
                if torch.cuda.is_current_stream_capturing():
                    self._captured_events.append(event)
            self._reader_streams.clear()

# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.

import logging
from abc import ABC, abstractmethod
from collections import defaultdict
from contextlib import contextmanager
from typing import Callable, Dict, List, Optional, Tuple

import torch
from torch.autograd import Variable

from megatron.core.utils import (
    get_pg_rank,
    get_pg_size,
    log_single_rank,
    make_viewless_tensor,
    nvtx_range_pop,
    nvtx_range_push,
)

logger = logging.getLogger(__name__)


class DeferredReleaseRegistry:
    """Singleton registry for deferred tensor memory release during CUDA graph capture.

    **WARNING**: due to the lack of "pytorch allocation stream" detection,
    we only assume a 2-stream (comm vs. comp) setup
    
    TODO: currently DeepEP's internal record_stream (on its own streams) might still get leaked
    Need further testing and consider adding this protection inside DeepEP too. 
    Hybrid EP currently should work fine

    During CUDA graph capture, record_stream() causes deferred frees in PyTorch's caching
    allocator that never resolve (because cudaEventQuery never returns True during capture),
    inflating the private pool memory. This registry replaces record_stream() + resize_(0)
    with a manual buffering scheme:

    1. Instead of record_stream + resize_(0), tensors are registered here with their
       associated event and the stream on which they were last consumed
       (producing_stream).
    2. When a ScheduleNode enters stream_acquire_context on a DIFFERENT stream and
       waits on the same event, it is guaranteed that producing_stream's work is
       complete. At that point, the registry drains matching tensors by calling
       resize_(0).
    3. This avoids record_stream entirely during capture while preserving the same
       memory reclamation timing as eager mode.

    The registry is keyed by (event_id, producing_stream_id). When a node on
    stream S waits on event E, it drains all entries keyed with event E whose
    producing_stream != S (i.e., tensors that were used on the OTHER stream and
    are now safe to free).

    Only active during CUDA graph capture; in eager mode, the original
    record_stream + resize_(0) path is used unchanged.

    Note: We do not track the true allocation stream of each tensor (PyTorch's
    caching allocator stores ``block->stream`` internally but does not expose it
    via any public Python API). Instead we record the *producing* node's stream
    — the stream that last wrote to the tensor. The drain logic relies on the
    combined_1f1b schedule using exactly two streams (comp and comm) that
    alternate via a shared event. A runtime check enforces this two-stream
    invariant: if more than two distinct streams are observed, an exception is
    raised immediately to prevent silent correctness issues.
    """

    _MAX_STREAMS = 2

    _instance = None

    def __init__(self):
        # Key: (id(event), cuda_stream_ptr) -> List[torch.Tensor]
        # cuda_stream_ptr is stream.cuda_stream (the raw cudaStream_t integer),
        # NOT id(stream), because multiple Python Stream objects can wrap the
        # same underlying CUDA stream.
        self._registry: Dict[Tuple[int, int], List[torch.Tensor]] = defaultdict(list)
        # Track all distinct cuda_stream_ptr values seen between drain_all() calls.
        self._seen_streams: Dict[int, torch.cuda.Stream] = {}

    @classmethod
    def get_instance(cls) -> "DeferredReleaseRegistry":
        """Get the singleton instance, creating it if necessary.

        Note: reset() is intentionally not called on exception paths.
        This registry is only active during CUDA graph capture, and an
        exception mid-capture leaves the graph itself in an unrecoverable
        state — the stale registry is dominated by that larger failure.
        drain_all() clears the registry at the end of every successful
        capture pass.
        """
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    @staticmethod
    def _stream_key(stream: torch.cuda.Stream) -> int:
        """Return the raw cudaStream_t pointer as the identity key for a stream.

        Using stream.cuda_stream instead of id(stream) because multiple Python
        Stream objects can wrap the same underlying CUDA stream (e.g. when
        streams are lazily constructed via a Callable per microbatch).
        """
        return stream.cuda_stream

    def _check_stream(self, stream: torch.cuda.Stream) -> None:
        """Record a stream and assert the two-stream invariant."""
        stream_id = self._stream_key(stream)
        if stream_id not in self._seen_streams:
            self._seen_streams[stream_id] = stream
            if len(self._seen_streams) > self._MAX_STREAMS:
                stream_reprs = [f"cuda_stream={sid}" for sid in self._seen_streams]
                raise RuntimeError(
                    f"DeferredReleaseRegistry: expected at most {self._MAX_STREAMS} "
                    f"distinct streams, but observed {len(self._seen_streams)}. "
                    f"Streams: [{', '.join(stream_reprs)}]. "
                    f"The deferred-release scheme assumes a two-stream "
                    f"(comp/comm) alternating schedule. If more streams are "
                    f"needed, the drain logic must be generalized."
                )

    def defer_release(
        self,
        tensors: List[torch.Tensor],
        event: torch.cuda.Event,
        producing_stream: torch.cuda.Stream,
    ):
        """Register tensors for deferred release.

        Args:
            tensors: List of tensors whose storage should be freed later.
            event: The CUDA event that will be recorded on producing_stream
                after the tensors' last use. A subsequent event.wait on another
                stream guarantees safety.
            producing_stream: The stream of the node that last used (consumed)
                these tensors. The tensors become safe to free once another
                stream waits on the event.
        """
        self._check_stream(producing_stream)
        key = (id(event), self._stream_key(producing_stream))
        self._registry[key].extend(tensors)

    def drain(self, event: torch.cuda.Event, waiting_stream: torch.cuda.Stream):
        """Free all deferred tensors that are now safe, given that waiting_stream
        has just waited on the given event.

        This should be called INSIDE stream_acquire_context, after event.wait(stream),
        while the stream context is active. It frees tensors whose producing_stream is
        different from waiting_stream (meaning the wait guarantees their work is done).

        Args:
            event: The event that was just waited on.
            waiting_stream: The stream that performed the wait (self.stream of the node).
        """
        self._check_stream(waiting_stream)
        event_id = id(event)
        waiting_key = self._stream_key(waiting_stream)
        # Collect keys to drain: same event, but producing_stream != waiting_stream
        keys_to_drain = [
            key for key in self._registry if key[0] == event_id and key[1] != waiting_key
        ]
        # NOTE: resize_(0) returns storage to the CUDA caching allocator.
        # The allocator tracks the *allocator-stream* (the stream that last
        # used the tensor), NOT the PyTorch stream context, so wrapping in
        # `with torch.cuda.stream(waiting_stream)` does not change when the
        # memory becomes reusable.  We keep the stream context purely for
        # semantic clarity: these deallocations logically belong to the
        # waiting stream, which has already synchronized with the producer
        # via event.wait().
        with torch.cuda.stream(waiting_stream):
            for key in keys_to_drain:
                tensors = self._registry.pop(key)
                for t in tensors:
                    t.untyped_storage().resize_(0)

    def drain_all(self):
        """Free ALL deferred tensors unconditionally.

        Called at synchronization points where all streams have been joined
        (e.g., at the end of TransformerModelChunkSchedulePlan.run() after
        wait_current_stream on both schedule plans). At that point, all GPU
        work is guaranteed complete and all deferred tensors are safe to free.

        Also resets the seen-streams record so the two-stream invariant is
        checked afresh for the next iteration / capture.
        """
        stream = torch.cuda.current_stream()
        with torch.cuda.stream(stream):
            for key in list(self._registry.keys()):
                tensors = self._registry.pop(key)
                for t in tensors:
                    t.untyped_storage().resize_(0)
        self._seen_streams.clear()

    def clear(self):
        """Clear all deferred tensors without freeing. Used for cleanup/reset."""
        self._registry.clear()
        self._seen_streams.clear()

    @classmethod
    def reset(cls):
        """Reset the singleton instance."""
        if cls._instance is not None:
            cls._instance.clear()
            cls._instance = None


def is_pp_first_stage(pp_group: torch.distributed.ProcessGroup):
    """Return True if in the first pipeline model-parallel stage, False otherwise."""
    return get_pg_rank(pp_group) == 0


def is_pp_last_stage(pp_group: torch.distributed.ProcessGroup):
    """Return True if in the last pipeline-model-parallel stage, False otherwise."""
    return get_pg_rank(pp_group) == (get_pg_size(pp_group) - 1)


def is_vp_first_stage(vp_stage: int, vp_size: int | None):
    """Return True if in the first virtual pipeline model-parallel stage, False otherwise."""
    if vp_size is None or vp_size <= 1:
        assert vp_stage is None or vp_stage == 0, (
            f"Expected vp_stage to be 0 or None when vp_size is <= 1 or None, "
            f"but got vp_stage={vp_stage} and vp_size={vp_size}"
        )
        return True
    return vp_stage == 0


def is_vp_last_stage(vp_stage: int, vp_size: int | None):
    """Return True if in the last virtual pipeline model-parallel stage, False otherwise."""
    if vp_size is None or vp_size <= 1:
        assert vp_stage is None or vp_stage == 0, (
            f"Expected vp_stage to be 0 or None when vp_size is <= 1 or None, "
            f"but got vp_stage={vp_stage} and vp_size={vp_size}"
        )
        return True
    return vp_stage == (vp_size - 1)


def get_pp_first_rank(pp_group: torch.distributed.ProcessGroup):
    """Return the global rank of the first rank in the pipeline parallel group."""
    pp_ranks = torch.distributed.get_process_group_ranks(pp_group)
    return pp_ranks[0]


def get_pp_last_rank(pp_group: torch.distributed.ProcessGroup):
    """Return the global rank of the last rank in the pipeline parallel group."""
    pp_ranks = torch.distributed.get_process_group_ranks(pp_group)
    return pp_ranks[-1]


def get_pp_next_rank(pp_group: torch.distributed.ProcessGroup):
    """Return the global rank of the next rank in the pipeline parallel group, or None if last
    stage."""
    if is_pp_last_stage(pp_group):
        return None
    current_rank_in_group = get_pg_rank(pp_group)
    pp_ranks = torch.distributed.get_process_group_ranks(pp_group)
    return pp_ranks[current_rank_in_group + 1]


def get_pp_prev_rank(pp_group: torch.distributed.ProcessGroup):
    """Return the global rank of the previous rank in the pipeline parallel group, or None if
    first stage."""
    if is_pp_first_stage(pp_group):
        return None
    current_rank_in_group = get_pg_rank(pp_group)
    pp_ranks = torch.distributed.get_process_group_ranks(pp_group)
    return pp_ranks[current_rank_in_group - 1]


def make_viewless(e):
    """Make_viewless util func"""
    e = make_viewless_tensor(inp=e, requires_grad=e.requires_grad, keep_graph=True)
    return e


def set_ideal_affinity_for_current_gpu():
    """Set CPU affinity for the current GPU to optimize host-device transfers."""
    import uuid

    try:
        import cuda.bindings.driver as cuda_driver
        import cuda.bindings.runtime as cuda_runtime
    except:
        try:
            import cuda.cuda as cuda_driver
            import cuda.cudart as cuda_runtime
        except:
            raise RuntimeError("Please install cuda-python to enable GPU affinity setting")
    import pynvml

    # Get current CUDA device ID
    err, device_id = cuda_runtime.cudaGetDevice()
    assert err == cuda_runtime.cudaError_t.cudaSuccess
    # Get device UUID
    err, device_uuid = cuda_driver.cuDeviceGetUuid(device_id)
    assert err == cuda_driver.CUresult.CUDA_SUCCESS
    # Set CPU affinity based on GPU's NUMA node
    pynvml.nvmlInit()
    handle = pynvml.nvmlDeviceGetHandleByUUID("GPU-" + str(uuid.UUID(bytes=device_uuid.bytes)))
    pynvml.nvmlDeviceSetCpuAffinity(handle)

    log_single_rank(
        logger,
        logging.WARNING,
        f"Set CPU affinity for all GPUs for optimal host-device transfer performance",
    )


class NoopScheduleNode:
    """A placeholder node in the computation graph that simply passes through inputs and outputs.

    This class is used as a no-op node in the scheduling system when a real computation node
    is not needed but the interface must be maintained (e.g., dense layer doesn't need
    moe_dispatch and moe_combine). It simply returns its inputs unchanged
    in both forward and backward passes.
    """

    def forward(self, inputs):
        """Passes through inputs unchanged in the forward pass."""
        return inputs

    def backward(self, outgrads):
        """Passes through gradients unchanged in the backward pass."""
        return outgrads


class ScheduleNode:
    """Base node for fine-grained scheduling.

    This class represents a computational node in the pipeline schedule.
    It handles the execution of forward and backward operations on a stream.
    """

    def __init__(
        self,
        forward_func: Callable,
        stream: torch.cuda.Stream,
        event: torch.cuda.Event,
        backward_func: Optional[Callable] = None,
        free_input: bool = False,
        name: str = "schedule_node",
    ):
        """Initialize a schedule node.

        Args:
            forward_func (callable): Function to execute during the forward pass.
            stream (Callable): Func that returns CUDA stream for computation.
                This can be either a 'compute' stream or a 'communicate' stream.
                - 'compute' stream: Used for computational nodes like attention and experts.
                - 'communicate' stream: Used for nodes that handle token communication,
                  such as token dispatch and combine operations in MoE layers.
            event (torch.cuda.Event): The CUDA event used for synchronization. Each
                microbatch within a model chunk shares the same event, which is used
                to manage dependencies between nodes operating on different streams.
            backward_func (callable, optional): Function for the backward pass.
            free_input (bool): Flag to indicate if the input should be freed after the
                forward pass.
            name (str): Name of the node for debugging purposes.
        """
        self.name = name
        self.forward_func = forward_func
        self.backward_func = backward_func if backward_func else self.default_backward_func
        self.stream = stream
        self.event = event
        self.free_input = free_input
        self.inputs = None
        self.outputs = None
        self.delay_grads_release = False
        self.manual_release_grads = False

    def default_backward_func(self, outputs, output_grad):
        """Default backward function"""
        Variable._execution_engine.run_backward(
            tensors=outputs,
            grad_tensors=output_grad,
            keep_graph=False,
            create_graph=False,
            inputs=tuple(),
            allow_unreachable=True,
            accumulate_grad=True,
        )
        return output_grad

    def forward(self, inputs=()):
        """Schedule node forward"""
        if not isinstance(inputs, tuple):
            inputs = (inputs,)
        return self._forward(*inputs)

    def _forward(self, *inputs):
        # Lazy initialization of stream
        if isinstance(self.stream, Callable):
            self.stream = self.stream()
        with self.stream_acquire_context(f"{self.name} forward"):
            self.inputs = [make_viewless(e).detach() if e is not None else None for e in inputs]
            for i, input in enumerate(self.inputs):
                if input is not None:
                    input.requires_grad = inputs[i].requires_grad

            data = tuple(self.inputs)
            data = self.forward_func(*data)

            if not isinstance(data, tuple):
                data = make_viewless(data)
            else:
                data = tuple([make_viewless(e) if isinstance(e, torch.Tensor) else e for e in data])

            self.output = data

        # Immediately frees input tensors after they are used for nodes
        # where inputs are no longer needed after computation.
        if self.free_input:
            if torch.cuda.is_current_stream_capturing():
                # During CUDA graph capture, record_stream causes deferred frees that
                # never resolve, inflating the private pool. Instead, defer the release
                # to the next node on the OTHER stream that waits on our event.
                deferred = [inp for inp in inputs if inp is not None]
                if deferred:
                    DeferredReleaseRegistry.get_instance().defer_release(
                        deferred, self.event, self.stream
                    )
            else:
                for input in inputs:
                    if input is not None:
                        input.record_stream(self.stream)
                        input.untyped_storage().resize_(0)

        return self.output

    def get_output(self):
        """Get the forward output"""
        return self.output

    def backward(self, output_grad):
        """Schedule node backward"""
        if not isinstance(output_grad, tuple):
            output_grad = (output_grad,)
        return self._backward(*output_grad)

    def _backward(self, *output_grad):
        # Lazy initialization of stream
        if isinstance(self.stream, Callable):
            self.stream = self.stream()
        with self.stream_acquire_context(f"{self.name} backward"):
            outputs = self.output
            if not isinstance(outputs, tuple):
                outputs = (outputs,)
            assert len(outputs) == len(output_grad), (
                f"{len(outputs)} of {type(outputs[0])} is not equal to "
                f"{len(output_grad)} of {type(output_grad[0])}"
            )
            output_grad = self.backward_func(outputs, output_grad)

        # output_grad maybe from another stream
        if output_grad:
            if torch.cuda.is_current_stream_capturing():
                # Note: unlike the eager path, we do NOT check manual_release_grads here.  
                # That flag suppresses resize_(0) for per-scope CG (e.g. CudaGraphScope.attn),  
                # but during full-iteration capture we must reclaim memory in the private pool. 
                if not self.delay_grads_release:
                    deferred = [g for g in output_grad if g is not None]
                    if deferred:
                        DeferredReleaseRegistry.get_instance().defer_release(
                            deferred, self.event, self.stream
                        )
            else:
                for g in output_grad:
                    if g is not None:
                        g.record_stream(self.stream)
                        # Manually trigger the memory release of dgrad tensor
                        # to avoid delayed garbage collection. If
                        # delay_grads_release is True, dgrad is last used in
                        # wgrad compute and skip the release here.
                        if self.manual_release_grads and not self.delay_grads_release:
                            g.untyped_storage().resize_(0)

        grads = self.get_grad()
        self._release_state()

        return grads

    def get_grad(self):
        """Get the grad of inputs"""
        grad = tuple([e.grad if e is not None else None for e in self.inputs])
        # multiple in, multiple out
        if len(grad) == 1:
            grad = grad[0]
        return grad

    @contextmanager
    def stream_acquire_context(self, name=None):
        """Stream acquire context that handles event synchronization,
            NVTX profiling, and stream context.

        This context manager consolidates:
        1. Event wait/record for synchronization between streams
        2. Deferred tensor release (during CUDA graph capture)
        3. NVTX range for profiling (if name is provided)
        4. torch.cuda.stream context for execution on the specified stream

        Args:
            name: Optional name for NVTX range profiling
        """
        self.event.wait(self.stream)
        # During CUDA graph capture, drain any deferred tensors that are now safe
        # to free. After event.wait(self.stream), all work on the OTHER stream
        # that recorded this event is guaranteed complete, so tensors produced
        # on that other stream can be safely freed.
        if torch.cuda.is_current_stream_capturing():
            DeferredReleaseRegistry.get_instance().drain(self.event, self.stream)
        if name:
            nvtx_range_push(name)
        try:
            with torch.cuda.stream(self.stream):
                yield
        finally:
            if name:
                nvtx_range_pop(name)
            self.event.record(self.stream)

    def _release_state(self):
        """Clear the state of the node"""
        self.inputs = None
        self.output = None
        del self.forward_func
        del self.backward_func


class AbstractSchedulePlan(ABC):
    """To use combined 1f1b, model must implement build_schedule_plan while take the same
    signature as model forward but return an instance of AbstractSchedulePlan"""

    @staticmethod
    @abstractmethod
    def run(
        f_schedule_plan,
        b_schedule_plan,
        grad=None,
        pre_forward=None,
        pre_backward=None,
        post_forward=None,
        post_backward=None,
    ):
        """run() is the protocol between our schedule logic and model, which is used to schedule
        the forward and backward schedule plans for the models.
        """
        ...


_USE_DYNAMIC_COMP_STREAM = None
_COMP_STREAM = None
_COMM_STREAM = None


def set_streams(comm_stream=None, high_priority=False):
    """Set the stream for communication operations."""
    global _COMM_STREAM

    # Set communication stream
    if _COMM_STREAM is None:
        if comm_stream is None:
            if high_priority:
                _, high = torch.cuda.Stream.priority_range()
                comm_stream = torch.cuda.Stream(device="cuda", priority=high)
            else:
                comm_stream = torch.cuda.Stream(device="cuda")
        _COMM_STREAM = comm_stream


def get_comp_stream():
    """Get the stream for computation"""
    return torch.cuda.current_stream()


def get_comm_stream():
    """Get the stream for communication"""
    global _COMM_STREAM
    return _COMM_STREAM

# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Utility classes and functions shared by the MoE experts offloading path.

Contents:

1) ``ExpertsWgradScheduler``: manages the scheduling of weight gradient computations for
   MoE experts, allowing wgrad computation to be delayed so that GPU computation and
   CPU-GPU communication interleave better.
2) ``StreamManager``: owns the CUDA streams and the stream-ordering helpers used to
   overlap expert-weight H2D transfers with compute.
3) ``build_offloading_expert_sharded_tensor``: builds the ``ShardedTensor`` for a single
   offloaded expert weight so checkpoints match the non-offloaded layouts.
"""

from __future__ import annotations

import queue
from typing import Any, Callable

import torch

from megatron.core.dist_checkpointing.mapping import ShardedTensor


class ExpertsWgradScheduler:
    """FIFO queue of deferred expert wgrad computations.

    When ``delay_wgrad_compute`` is set, the offloading autograd function registers each
    wgrad closure here instead of running it inline, and ``backward_dw()`` drains the queue
    after the dgrad pass has completed.
    """

    def __init__(self, delay_wgrad_compute: bool = False) -> None:
        self.delay_wgrad_compute = delay_wgrad_compute
        self.queue: queue.Queue = queue.Queue()

    def register(self, grad_func: Callable[..., Any], *grad_parms: Any) -> None:
        """Queue ``grad_func(*grad_parms)`` for later execution, if wgrad is delayed."""
        if self.delay_wgrad_compute:
            self.queue.put((grad_func, grad_parms))

    def pop_callback(self) -> Any:
        """Run the oldest queued wgrad closure, or do nothing if the queue is empty."""
        if self.queue.qsize() > 0 and self.delay_wgrad_compute:
            grad_func, grad_parms = self.queue.get()
            return grad_func(*grad_parms)
        else:
            # If there is no token assigned to the expert in this MoE layer,
            # then there will be case that the wgrad compute is not registered
            return


def release(t: torch.Tensor) -> None:
    """Helper function to release tensors that are no longer needed to save memory."""
    t.untyped_storage().resize_(0)


class StreamManager:
    """Manage CUDA streams and events shared by MoE offloading paths."""

    _instance = None

    def __init__(self, num_h2d_streams: int, num_compute_streams: int = 4) -> None:
        self.num_compute_streams = num_compute_streams
        self.num_h2d_streams = num_h2d_streams
        self.h2d_streams = [torch.cuda.Stream() for _ in range(num_h2d_streams)]
        self.compute_streams = [torch.cuda.Stream() for _ in range(self.num_compute_streams)]
        self.compute_cuda_streams = [stream.cuda_stream for stream in self.compute_streams]

        # Dedicated copy streams for activation offload D2H/H2D.
        self.act_d2h_stream = torch.cuda.Stream()
        self.act_h2d_stream = torch.cuda.Stream()

    @classmethod
    def get_instance(cls, num_h2d_streams: int = 2, num_compute_streams: int = 4) -> StreamManager:
        """Return the process-wide singleton, creating it on first use."""
        if cls._instance is None:
            cls._instance = StreamManager(num_h2d_streams, num_compute_streams)
        return cls._instance

    def get_h2d_stream(self, idx: int) -> torch.cuda.Stream:
        """Return the ``idx``-th host-to-device copy stream."""
        return self.h2d_streams[idx]

    def get_compute_streams(self) -> list[int]:
        """Return the raw ``cudaStream_t`` handles of the compute streams."""
        return self.compute_cuda_streams

    def get_compute_stream_objects(self) -> list[torch.cuda.Stream]:
        """Return the compute streams as ``torch.cuda.Stream`` objects."""
        return self.compute_streams

    def get_launch_streams(self) -> list[torch.cuda.Stream]:
        """Return the streams the caller may have launched this module from.

        This is the current stream, plus the default stream when they differ: with virtual
        pipeline parallelism a model chunk can execute on a non-default current stream, and
        both need to participate in the ordering below.
        """
        # VPP can execute a model chunk on a non-default current stream.
        current_stream = torch.cuda.current_stream()
        default_stream = torch.cuda.default_stream()
        if current_stream.cuda_stream == default_stream.cuda_stream:
            return [current_stream]
        return [current_stream, default_stream]

    def launch_streams_wait_compute_streams(self) -> None:
        """Join the compute streams back into the launch streams."""
        launch_streams = self.get_launch_streams()
        for i in range(self.num_compute_streams):
            for launch_stream in launch_streams:
                launch_stream.wait_stream(self.compute_streams[i])

    def default_stream_wait_h2d_stream(self, idx: int) -> None:
        """Make the default stream wait for the ``idx``-th H2D copy stream."""
        torch.cuda.default_stream().wait_stream(self.get_h2d_stream(idx))

    def compute_streams_wait_launch_streams(self) -> None:
        """Fork the compute streams off the launch streams."""
        launch_streams = self.get_launch_streams()
        for i in range(self.num_compute_streams):
            for launch_stream in launch_streams:
                self.compute_streams[i].wait_stream(launch_stream)

    def h2d_stream_wait_consumer_streams(self, idx: int) -> None:
        """Make the ``idx``-th H2D stream wait for every consumer of its staging buffer.

        This is what makes the staging buffer safe to overwrite with the next chunk.
        """
        h2d_stream = self.get_h2d_stream(idx)
        for launch_stream in self.get_launch_streams():
            h2d_stream.wait_stream(launch_stream)
        for i in range(self.num_compute_streams):
            h2d_stream.wait_stream(self.compute_streams[i])

    def compute_streams_wait_h2d_stream(self, idx: int) -> None:
        """Make the compute streams wait for the ``idx``-th H2D copy stream."""
        h2d_stream = self.get_h2d_stream(idx)
        for i in range(self.num_compute_streams):
            self.compute_streams[i].wait_stream(h2d_stream)

    def consumer_streams_wait_event(self, event: torch.cuda.Event) -> None:
        """Make the launch and compute streams wait on ``event``."""
        for launch_stream in self.get_launch_streams():
            launch_stream.wait_event(event)
        for i in range(self.num_compute_streams):
            self.compute_streams[i].wait_event(event)

    def h2d_stream_wait_default_stream(self, idx: int) -> None:
        """Make the ``idx``-th H2D copy stream wait for the default stream."""
        self.get_h2d_stream(idx).wait_stream(torch.cuda.default_stream())

    def act_d2h_stream_wait_producers(self) -> None:
        """Make the activation-offload D2H stream wait for activation producers."""
        for launch_stream in self.get_launch_streams():
            self.act_d2h_stream.wait_stream(launch_stream)
        for i in range(self.num_compute_streams):
            self.act_d2h_stream.wait_stream(self.compute_streams[i])

    def consumer_streams_wait_act_reload(self, h2d_done_event: torch.cuda.Event) -> None:
        """Make backward consumer streams wait until activation reload H2D completes."""
        self.consumer_streams_wait_event(h2d_done_event)


_dummy_wgrads = {}


def get_dummy_wgrad(
    shape: list[int], dtype: torch.dtype, device: torch.device | int, zero: bool = False
) -> torch.Tensor:
    """Returns a dummy tensor of given shape."""
    global _dummy_wgrads
    wgard_key = (*shape, dtype)
    if wgard_key not in _dummy_wgrads:
        _dummy_wgrads[wgard_key] = torch.empty(
            shape, dtype=dtype, device=device, requires_grad=False
        )
    if zero:
        _dummy_wgrads[wgard_key].fill_(0)
    return _dummy_wgrads[wgard_key].detach()


def build_offloading_expert_sharded_tensor(
    weight_slice: torch.Tensor,
    prefix: str,
    weight_name: str,
    global_expert_idx: int,
    *,
    sharded_offsets: tuple,
    num_global_experts: int,
    replica_id: tuple[int, int, int],
    singleton_local_shards: bool,
    transpose: bool,
) -> ShardedTensor:
    """Build the ShardedTensor for a single offloaded expert weight.

    Produces the *same* on-disk representation for both OffloadingExpertsMLP
    variants so their checkpoints are interchangeable:

    - bf16 variant: per-expert params stored as ``(in, out)`` -> ``transpose=False``
    - inplace-fp8 variant: fused master stored as ``(out, in)`` (NT layout)
      -> ``transpose=True`` to land on the same ``(in, out)`` checkpoint layout.

    The expert is expressed as a *prepended* axis fragment exactly like
    ``SequentialMLP``/``TEGroupedMLP`` (see ``apply_swiglu_sharded_factory``),
    i.e. ``prepend_axis_num == len(offsets)``, which is what makes the expert
    dimension compose cleanly with any pipeline/tensor ``sharded_offsets``.

    Args:
        weight_slice (torch.Tensor): one expert's 2D weight. ``(in, out)`` when
            ``transpose=False`` else ``(out, in)``.
        prefix (str): module prefix for the on-disk key.
        weight_name (str): ``'weight1'`` or ``'weight2'``.
        global_expert_idx (int): this expert's index in the global expert range.
        sharded_offsets (tuple): offsets inherited from parent modules (PP, ...).
        num_global_experts (int): total experts across the expert-parallel group.
        replica_id: ShardedTensor replica id (PP, TP, DP).
        singleton_local_shards (bool): when True each expert is saved under its own
            global key with no expert sharding axis.
        transpose (bool): transpose ``(out, in) -> (in, out)`` before saving.
    """
    data = weight_slice.transpose(0, 1).contiguous() if transpose else weight_slice
    if singleton_local_shards:
        key = f'{prefix}experts.{global_expert_idx}.{weight_name}'
        offsets = sharded_offsets
    else:
        key = f'{prefix}experts.{weight_name}'
        offsets = (*sharded_offsets, (len(sharded_offsets), global_expert_idx, num_global_experts))
    return ShardedTensor.from_rank_offsets(
        key, data, *offsets, replica_id=replica_id, prepend_axis_num=len(offsets)
    )

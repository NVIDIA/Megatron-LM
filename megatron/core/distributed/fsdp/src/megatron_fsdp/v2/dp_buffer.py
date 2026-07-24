# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

import enum
from typing import Dict, List, Optional

import torch
from torch.distributed.tensor import DeviceMesh

from .allocator import BucketAllocator, TemporaryBucketAllocator, _free_storage
from .utils import ParamGroupIdx


class Placement(enum.Enum):
    """Logical state of a DP buffer along one mesh dimension.

    A buffer stores two enum members ordered as ``[outer-DP, inner-DP]``.

    ``FLAT`` and ``DIRTY`` contain the same valid rank-owned shard. ``FLAT``
    has compact shard storage, while ``DIRTY`` keeps full-sized storage whose
    non-owned regions are invalid. ``PARTIAL`` is a local contribution pending
    reduction; it is not another form of ``DIRTY``.

    Supported data transitions are:

    - ``FLAT``/``DIRTY`` -> ``REPLICATE``: all-gather
    - ``PARTIAL`` -> ``REPLICATE``: all-reduce
    - ``PARTIAL`` -> ``FLAT``/``DIRTY``: reduce-scatter
    - ``REPLICATE`` -> ``FLAT``: retain the rank-owned shard
    - ``REPLICATE`` -> ``DIRTY``: update only the rank-owned shard
    - ``FLAT`` -> ``DIRTY``: place the shard into full-sized storage
    - ``DIRTY`` -> ``FLAT``: discard invalid full-sized storage
    """

    FLAT = "flat"
    REPLICATE = "replicate"
    PARTIAL = "partial"
    DIRTY = "dirty"


class DataParallelBuffer:
    """Manages a flat buffer that stores (a shard of) a group of parameters.

    On construction it builds its own BufferIndex describing the layout and
    shard ownership.  External callers interact via init_data / set_item /
    get_item only.
    """

    def __init__(
        self,
        params: List[torch.nn.Parameter],
        param_idx: Dict[torch.nn.Parameter, int],
        dtype: torch.dtype,
        device: torch.device,
        mesh: DeviceMesh,
        param_group_id: ParamGroupIdx,
        mp_policy,
        *,
        allocator: Optional[BucketAllocator] = None,
        buffer_role: str = "model_weight",
        gradient_scaling_factor: Optional[float] = None,
        chunk_size_factor: int = 1,
        sharding_strategy: str = "no_shard",
        outer_dp_sharding_strategy: str = "no_shard",
    ):
        # Keep BufferIndex's Placement import from forming a module-level cycle.
        from .buffer_index import BufferIndex

        assert mp_policy is not None, "DataParallelBuffer requires a mixed-precision policy"
        self.params = params
        self.param_idx = param_idx
        self.dtype = dtype
        self.device = device
        self.outer_dp_group = mesh.get_group(mesh_dim=0)
        self.inner_dp_group = mesh.get_group(mesh_dim=1)
        self.allocator = allocator if allocator is not None else TemporaryBucketAllocator()
        self.buffer_role = buffer_role
        self.alloc_key = (param_group_id, buffer_role)
        self.mp_policy = mp_policy
        self.grad_comm_dtype = mp_policy.grad_comm_dtype or dtype
        self._use_grad_comm_buffer = self.grad_comm_dtype != dtype

        def is_sharded_from_strategy(strategy: str) -> bool:
            if buffer_role in ("model_weight", "transpose_weight"):
                return strategy == "optim_grads_params"
            if buffer_role == "main_weight":
                return strategy != "no_shard"
            if buffer_role == "main_grad":
                return strategy in ("optim_grads", "optim_grads_params")
            raise ValueError(f"Unsupported data-parallel buffer role: {buffer_role}")

        self.outer_sharded = is_sharded_from_strategy(outer_dp_sharding_strategy)
        self.inner_sharded = is_sharded_from_strategy(sharding_strategy)
        self.storage_placements: list[Placement] = [
            Placement.FLAT if sharded else Placement.REPLICATE
            for sharded in (self.outer_sharded, self.inner_sharded)
        ]
        self.placements: list[Placement] = self.storage_placements.copy()
        self.sharding_strategy = sharding_strategy
        self.outer_dp_sharding_strategy = outer_dp_sharding_strategy
        self.gradient_scaling_factor = gradient_scaling_factor

        # Always build layout with logical shapes and shared chunk_size_factor
        # so that all buffers share the same proportional item-offset mapping.
        _logical_shapes = [p.shape for p in params]
        self.buffer_index = BufferIndex(
            param_shapes=_logical_shapes,
            mesh=mesh,
            chunk_size_factor=chunk_size_factor,
            param_group_id=param_group_id,
        )

        # Compact NVFP4 weight buffers: scale all indices proportionally so
        # the buffer holds only the packed data without fragment-binning waste.
        if buffer_role in ("model_weight", "transpose_weight") and any(
            mp_policy.is_nvfp4_param(p) for p in params
        ):
            compact_shapes = mp_policy.get_param_storage_shapes(params)
            self.buffer_index.compact(0.5, compact_shapes)

        # Dirty has larger physical storage, but buffers are never initialized as Dirty.
        self.data_size = self.buffer_index._get_shard_meta(self.storage_placements).size

        self.data: Optional[torch.Tensor] = None
        self._unsharded_buffer: Optional[torch.Tensor] = None

    # ------------------------------------------------------------------ #
    #  Public API
    # ------------------------------------------------------------------ #

    def init_data(self, data: torch.Tensor) -> None:
        """Bind an externally allocated tensor as the persistent storage."""
        assert data.dtype == self.dtype, f"dtype mismatch: {data.dtype} vs {self.dtype}"
        assert data.numel() == self.data_size, f"size mismatch: {data.numel()} vs {self.data_size}"
        self.data = data

    # ------------------------------------------------------------------ #
    #  CPU offload
    # ------------------------------------------------------------------ #

    def _is_on_cpu(self) -> bool:
        """True if ``self.data`` is resident on CPU."""
        return self.data is not None and self.data.device.type == "cpu"

    def _ensure_data_on_gpu(self) -> bool:
        """Move ``self.data`` to GPU if currently on CPU.

        Returns True if a move happened (caller must rebuild dist views).
        """
        if not self._is_on_cpu():
            return False
        self.data = self.data.to(self.device, non_blocking=True)
        return True

    def _move_data_to(
        self, target_device: torch.device, pin_memory: bool = False, non_blocking: bool = True
    ) -> None:
        """Move ``self.data`` to *target_device*, optionally using pinned memory.

        Caller must call ``ParameterGroup._rebuild_dist_views()`` afterwards
        because ``dist_params._local_tensor`` views share ``self.data`` Storage.
        """
        if self.data is None or self.data.device == target_device:
            return
        if target_device.type == "cpu" and pin_memory:
            cpu_data = torch.empty(self.data.shape, dtype=self.data.dtype, pin_memory=True)
            cpu_data.copy_(self.data, non_blocking=non_blocking)
            _free_storage(self.data)
            self.data = cpu_data
        else:
            self.data = self.data.to(target_device, non_blocking=non_blocking)

    @torch.no_grad()
    def set_item(
        self, item_id: int, item_data: torch.Tensor, *, placements: Optional[list[Placement]] = None
    ) -> None:
        """Write a parameter tensor into the corresponding region of the buffer."""
        requested_placements = placements if placements is not None else self.placements
        assert not any(
            placement is Placement.DIRTY for placement in requested_placements
        ), "set_item does not support Dirty placements"
        source_slice, local_slice = self.buffer_index.local_slice_for(
            self.buffer_index._get_item_global_range(item_id),
            requested_placements,
            self.storage_placements,
        )
        if source_slice is None or local_slice is None:
            return
        self.data[local_slice].copy_(item_data.flatten()[source_slice])

    def get_item(
        self, item_id: int, *, placements: Optional[list[Placement]] = None
    ) -> torch.Tensor:
        """Read a parameter tensor (or its shard) from the buffer."""
        requested_placements = placements if placements is not None else self.placements
        assert not any(
            placement is Placement.DIRTY for placement in requested_placements
        ), "get_item does not support Dirty placements"
        _, local_slice = self.buffer_index.local_slice_for(
            self.buffer_index._get_item_global_range(item_id),
            requested_placements,
            self.storage_placements,
        )
        return self.data[:0] if local_slice is None else self.data[local_slice]

    def is_unsharded(self) -> bool:
        """Return whether this buffer currently has a full unsharded view."""
        return all(placement is Placement.REPLICATE for placement in self.placements)

    @torch.no_grad()
    def redistribute(
        self,
        target_placements: Optional[list[Placement]] = None,
        *,
        stream: Optional[torch.cuda.Stream] = None,
    ) -> torch.Tensor:
        """Redistribute one mesh axis and return the branch output."""
        if target_placements is None:
            target_placements = self.storage_placements
        assert len(target_placements) == 2

        changed_axis = None
        for axis, (source, target) in enumerate(zip(self.placements, target_placements)):
            if source == target:
                continue
            if changed_axis is not None:
                raise ValueError(
                    "redistribute supports changing only one placement axis per call: "
                    f"{self.placements} -> {target_placements}"
                )
            changed_axis = axis
        if changed_axis is None:
            return self.fetch_buffer(target_placements)

        current_stream = torch.cuda.current_stream()
        stream = stream or current_stream
        if stream != current_stream:
            stream.wait_stream(current_stream)

        source = self.placements[changed_axis]
        target = target_placements[changed_axis]
        input_buffer = self.fetch_buffer(self.placements)
        output = self.fetch_buffer(target_placements)
        group = self.outer_dp_group if changed_axis == 0 else self.inner_dp_group

        if source in (Placement.FLAT, Placement.DIRTY) and target is Placement.REPLICATE:
            with torch.cuda.stream(stream):
                torch.distributed.all_gather_into_tensor(output, input_buffer, group=group)
        elif source is Placement.PARTIAL:
            scaling_factor = self.gradient_scaling_factor
            scale_inner = changed_axis == 1 and scaling_factor not in (None, 1.0)
            prescale = scale_inner and (
                self.grad_comm_dtype == torch.bfloat16
                or torch.distributed.get_world_size(group) == 1
            )
            op = (
                torch.distributed.ReduceOp.SUM
                if not scale_inner or prescale
                else torch.distributed._make_nccl_premul_sum(scaling_factor)
            )

            comm_input = input_buffer
            if self._use_grad_comm_buffer:
                comm_input = self.allocator.allocate(
                    key=(self.alloc_key, "grad_reduce_input", changed_axis),
                    size=input_buffer.numel(),
                    dtype=self.grad_comm_dtype,
                    device=self.device,
                ).data
            if comm_input.is_cuda:
                comm_input.record_stream(stream)

            with torch.cuda.stream(stream):
                if comm_input is not input_buffer:
                    comm_input.copy_(input_buffer)
                if prescale:
                    comm_input.mul_(scaling_factor)
                if target is Placement.REPLICATE:
                    torch.distributed.all_reduce(comm_input, group=group, op=op)
                    output = comm_input
                else:
                    input_meta = self.buffer_index._get_shard_meta(self.placements)
                    output_meta = self.buffer_index._get_shard_meta(target_placements)
                    output_offset = output_meta.global_data_index - input_meta.global_data_index
                    output = comm_input[output_offset : output_offset + output_meta.size]
                    torch.distributed.reduce_scatter_tensor(
                        output=output, input=comm_input, group=group, op=op
                    )
        elif target in (Placement.DIRTY, Placement.PARTIAL):
            pass
        elif target is Placement.FLAT:
            if source is Placement.REPLICATE:
                self.release_unsharded_buffer()
        else:
            raise NotImplementedError(f"Unsupported placement transition: {source!r} -> {target!r}")

        self.placements[changed_axis] = target
        return output

    @torch.no_grad()
    def commit_comm_output(
        self,
        comm_output: torch.Tensor,
        changed_axis: int,
        *,
        stream: Optional[torch.cuda.Stream] = None,
        accumulate: bool = False,
    ) -> None:
        """Copy or accumulate a communication result into its target buffer."""
        output_buffer = self.fetch_buffer(self.placements)
        with torch.cuda.stream(stream or torch.cuda.current_stream()):
            if output_buffer.data_ptr() != comm_output.data_ptr():
                if accumulate:
                    output_buffer.add_(comm_output)
                else:
                    output_buffer.copy_(comm_output)
        if self._use_grad_comm_buffer:
            self.allocator.free((self.alloc_key, "grad_reduce_input", changed_axis))

    def bind_params(self, buffer: Optional[torch.Tensor] = None) -> None:
        """Bind parameters to a fully replicated buffer."""
        if buffer is None:
            assert self.is_unsharded(), "Cannot bind params from a sharded buffer"
            buffer = self.fetch_buffer(self.placements)
        assert buffer.numel() == self.buffer_index.bucket_meta.size, (
            f"Buffer size {buffer.numel()} does not match expected size "
            f"{self.buffer_index.bucket_meta.size}"
        )
        for p in self.params:
            item_id = self.param_idx[p]
            start, end = self.buffer_index._get_item_global_range(item_id)
            idx_shape = self.buffer_index.item_index_map[item_id].shape
            param_data = buffer[start:end].view(idx_shape)
            self.mp_policy.bind_unsharded_param(p, param_data, self.buffer_role)

    def release_unsharded_buffer(self) -> None:
        """Release the temporary full-sized buffer without changing placements."""
        self.allocator.free(self.alloc_key)
        self._unsharded_buffer = None

    def get_shard_view(self, placements: Optional[list[Placement]] = None) -> torch.Tensor:
        """Return a placement view inside the persistent data buffer."""
        assert self.data is not None, "DataParallelBuffer data not initialized"
        requested_placements = placements if placements is not None else self.placements
        _, local_slice = self.buffer_index.local_slice_for(
            (0, self.buffer_index.bucket_meta.size), requested_placements, self.storage_placements
        )
        return self.data[:0] if local_slice is None else self.data[local_slice]

    def fetch_buffer(self, placements: list[Placement]) -> torch.Tensor:
        """Return a buffer for placements, allocating temporary storage if needed.

        1. If placements match the storage placements, return self.data directly.
        2. If self.data is a known parent of the requested placements, return a
           view into self.data.
        3. Otherwise allocate/reuse the fully replicated temporary buffer and
           return either that full buffer or a view from it.

        Memory allocation always occurs on the caller stream for deterministic
        caching-allocator behaviour.
        """
        requested_meta = self.buffer_index._get_shard_meta(placements)
        if placements == self.storage_placements:
            assert self.data is not None, "DataParallelBuffer data not initialized"
            return self.data

        data_contains_requested = all(
            storage_placement is not Placement.FLAT or requested_placement is Placement.FLAT
            for storage_placement, requested_placement in zip(self.storage_placements, placements)
        )
        if data_contains_requested:
            return self.get_shard_view(placements)

        if self._unsharded_buffer is None:
            bucket = self.allocator.allocate(
                key=self.alloc_key,
                size=self.buffer_index.bucket_meta.size,
                dtype=self.dtype,
                device=self.device,
            )
            self._unsharded_buffer = bucket.data
        if all(placement is Placement.REPLICATE for placement in placements):
            return self._unsharded_buffer
        return self._unsharded_buffer[
            requested_meta.bucket_data_index : requested_meta.bucket_data_index
            + requested_meta.size
        ]

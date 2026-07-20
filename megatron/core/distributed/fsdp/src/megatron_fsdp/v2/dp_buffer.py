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

        def is_sharded_from_strategy(strategy: str) -> bool:
            if buffer_role in ("model_weight", "transpose_weight"):
                return strategy == "optim_grads_params"
            if buffer_role == "main_weight":
                return strategy != "no_shard"
            if buffer_role == "main_grad":
                return strategy in ("optim_grads", "optim_grads_params")
            raise ValueError(f"Unsupported data-parallel buffer role: {buffer_role}")

        self.storage_placements: list[Placement] = [
            Placement.FLAT
            if is_sharded_from_strategy(outer_dp_sharding_strategy)
            else Placement.REPLICATE,
            Placement.FLAT
            if is_sharded_from_strategy(sharding_strategy)
            else Placement.REPLICATE,
        ]
        self.placements: list[Placement] = self.storage_placements.copy()
        self.outer_sharded = self.storage_placements[0] is Placement.FLAT
        self.inner_sharded = self.storage_placements[1] is Placement.FLAT
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
        self,
        item_id: int,
        item_data: torch.Tensor,
        *,
        placements: Optional[list[Placement]] = None,
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
        **kwargs,
    ) -> None:
        """Redistribute to the target or persistent storage placements."""
        if target_placements is None:
            target_placements = self.storage_placements
        assert len(target_placements) == 2

        current_stream = torch.cuda.current_stream()
        stream = stream or current_stream
        if stream != current_stream:
            stream.wait_stream(current_stream)

        if self.placements == target_placements:
            return

        reduce_inner_first = (
            self.placements[1] is Placement.PARTIAL
            and target_placements[1] is not Placement.PARTIAL
        )
        comm_order = (1, 0) if reduce_inner_first else (0, 1)

        for current_dim in comm_order:
            source_placement = self.placements[current_dim]
            target_placement = target_placements[current_dim]
            if source_placement == target_placement:
                continue

            next_placements = self.placements.copy()
            next_placements[current_dim] = target_placement

            if (
                source_placement in (Placement.FLAT, Placement.DIRTY)
                and target_placement is Placement.REPLICATE
            ):
                self.all_gather(
                    target_placements=next_placements,
                    comm_dim=current_dim,
                    stream=stream,
                    **kwargs,
                )
            elif (
                source_placement is Placement.PARTIAL
                and target_placement is Placement.REPLICATE
            ):
                self.reduce_grad(
                    target_placements=next_placements,
                    comm_dim=current_dim,
                    stream=stream,
                    reduce_scatter=False,
                    **kwargs,
                )
            elif (
                source_placement is Placement.PARTIAL
                and target_placement in (Placement.FLAT, Placement.DIRTY)
            ):
                self.reduce_grad(
                    target_placements=next_placements,
                    comm_dim=current_dim,
                    stream=stream,
                    reduce_scatter=True,
                    **kwargs,
                )
            elif target_placement in (Placement.DIRTY, Placement.PARTIAL):
                pass
            elif target_placement is Placement.FLAT:
                self.reshard(
                    target_placements=next_placements,
                    comm_dim=current_dim,
                    stream=stream,
                    **kwargs,
                )
            else:
                raise NotImplementedError(
                    f"Unsupported placement transition: "
                    f"{source_placement!r} -> {target_placement!r}"
                )

            self.placements = next_placements

    @torch.no_grad()
    def all_gather(
        self,
        target_placements: list[Placement],
        *,
        comm_dim: int,
        stream: torch.cuda.Stream,
        **kwargs,
    ) -> torch.Tensor:
        """All-gather the selected mesh dimension into target placements."""
        source_placements = self.placements
        group = self.outer_dp_group if comm_dim == 0 else self.inner_dp_group
        input_buffer = self.fetch_buffer(source_placements)
        output_buffer = self.fetch_buffer(target_placements)
        if torch.distributed.get_world_size(group) == 1:
            with torch.cuda.stream(stream):
                if output_buffer.data_ptr() != input_buffer.data_ptr():
                    output_buffer.copy_(input_buffer)
        else:
            with torch.cuda.stream(stream):
                torch.distributed.all_gather_into_tensor(
                    output_tensor=output_buffer, input_tensor=input_buffer, group=group
                )

        if kwargs.get("bind_params", False) and all(
            placement is Placement.REPLICATE for placement in target_placements
        ):
            self._bind_buffer_to_params(output_buffer)
        return output_buffer

    def _bind_buffer_to_params(self, buffer: torch.Tensor) -> None:
        """Bind the given buffer to the params according to the layout."""
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

    @torch.no_grad()
    def reshard(
        self,
        *,
        comm_dim: Optional[int] = None,
        **kwargs,
    ) -> None:
        """Reshard replicated placements and release temporary full storage."""
        if comm_dim is not None:
            source_placement = self.placements[comm_dim]
            if source_placement is not Placement.REPLICATE:
                return
        self.release_unsharded_buffer()

    def release_unsharded_buffer(self) -> None:
        """Release the temporary full-sized buffer without changing placements."""
        self.allocator.free(self.alloc_key)
        self._unsharded_buffer = None

    def get_shard_view(
        self,
        placements: Optional[list[Placement]] = None,
    ) -> torch.Tensor:
        """Return a placement view inside the persistent data buffer."""
        assert self.data is not None, "DataParallelBuffer data not initialized"
        requested_placements = (
            placements if placements is not None else self.placements
        )
        _, local_slice = self.buffer_index.local_slice_for(
            (0, self.buffer_index.bucket_meta.size),
            requested_placements,
            self.storage_placements,
        )
        return self.data[:0] if local_slice is None else self.data[local_slice]

    def fetch_buffer(
        self,
        placements: list[Placement],
    ) -> torch.Tensor:
        """Return a buffer for placements, allocating temporary storage if needed.

        1. If placements match the storage placements, return self.data directly.
        2. If self.data is a known parent of the requested placements, return a
           view into self.data.
        3. Otherwise allocate/reuse the fully replicated temporary buffer and
           return either that full buffer or a view from it.

        Memory allocation always occurs on the caller stream for deterministic
        caching-allocator behaviour.
        """
        assert self.data is not None, "DataParallelBuffer data not initialized"
        requested_meta = self.buffer_index._get_shard_meta(placements)
        if placements == self.storage_placements:
            return self.data

        data_contains_requested = all(
            storage_placement is not Placement.FLAT
            or requested_placement is Placement.FLAT
            for storage_placement, requested_placement in zip(
                self.storage_placements, placements
            )
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
        if all(
            placement is Placement.REPLICATE
            for placement in placements
        ):
            return self._unsharded_buffer
        return self._unsharded_buffer[
            requested_meta.bucket_data_index : requested_meta.bucket_data_index
            + requested_meta.size
        ]

    @torch.no_grad()
    def reduce_grad(
        self,
        target_placements: list[Placement],
        *,
        comm_dim: int,
        reduce_scatter: bool,
        stream: torch.cuda.Stream,
        **kwargs,
    ) -> torch.Tensor:
        """Reduce a partial value into the requested target placements."""
        source_placements = self.placements
        input_buffer = self.fetch_buffer(source_placements)
        output_buffer = self.fetch_buffer(target_placements)
        group = self.outer_dp_group if comm_dim == 0 else self.inner_dp_group
        # Inner-DP shards accumulate across microbatches; outer-DP only reduces
        # the completed inner result and therefore always overwrites its output.
        reduce_inner = comm_dim == 1
        accumulate = kwargs.get("accumulate", False) and reduce_inner
        grad_comm_dtype = (
            kwargs.get("grad_comm_dtype")
            or self.mp_policy.grad_comm_dtype
            or self.dtype
        )

        # Scale exactly once, when reducing fresh full grads over inner-DP.
        # Outer-only reduce consumes an already-scaled inner-DP result.
        if not reduce_inner or self.gradient_scaling_factor in (None, 1.0):
            op = torch.distributed.ReduceOp.SUM
            prescale = False
        elif grad_comm_dtype != torch.bfloat16:
            op = torch.distributed._make_nccl_premul_sum(
                self.gradient_scaling_factor
            )
            prescale = False
        else:
            op = torch.distributed.ReduceOp.SUM
            prescale = True

        if torch.distributed.get_world_size(group) == 1:
            if input_buffer.is_cuda:
                input_buffer.record_stream(stream)
            with torch.cuda.stream(stream):
                # A singleton inner-DP group bypasses both NCCL premul-sum and the
                # BF16 prescale path above, so apply its scaling locally.
                if reduce_inner and self.gradient_scaling_factor not in (None, 1.0):
                    input_buffer.mul_(self.gradient_scaling_factor)
                if output_buffer.data_ptr() != input_buffer.data_ptr():
                    if accumulate:
                        output_buffer.add_(input_buffer)
                    else:
                        output_buffer.copy_(input_buffer)
            return output_buffer

        comm_input = input_buffer
        input_key = None
        if grad_comm_dtype != self.dtype:
            input_key = (
                self.alloc_key,
                "grad_reduce_input",
                comm_dim,
            )
            input_bucket = self.allocator.allocate(
                key=input_key, size=input_buffer.numel(), dtype=grad_comm_dtype, device=self.device
            )
            comm_input = input_bucket.data
            with torch.cuda.stream(stream):
                comm_input.copy_(input_buffer)
        if comm_input.is_cuda:
            comm_input.record_stream(stream)
        if prescale:
            with torch.cuda.stream(stream):
                comm_input.mul_(self.gradient_scaling_factor)

        if not reduce_scatter:
            with torch.cuda.stream(stream):
                torch.distributed.all_reduce(comm_input, group=group, op=op)
                if input_key is not None:
                    output_buffer.copy_(comm_input.to(self.dtype))
            if input_key is not None:
                self.allocator.free(input_key)
            return output_buffer

        input_meta = self.buffer_index._get_shard_meta(source_placements)
        output_meta = self.buffer_index._get_shard_meta(target_placements)
        output_offset = output_meta.global_data_index - input_meta.global_data_index
        # Stage RS output in the input buffer slice; avoids untraced temp keys in TracePool.
        comm_output = comm_input[
            output_offset : output_offset + output_buffer.numel()
        ]

        with torch.cuda.stream(stream):
            torch.distributed.reduce_scatter_tensor(
                output=comm_output, input=comm_input, group=group, op=op
            )
            if output_buffer.data_ptr() != comm_output.data_ptr():
                if accumulate:
                    output_buffer.add_(comm_output)
                else:
                    output_buffer.copy_(comm_output)
        if input_key is not None:
            self.allocator.free(input_key)
        return output_buffer
